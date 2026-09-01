#!/usr/bin/env python3
"""
cli.py — Command-line interface for the RAG pipeline
=====================================================

Usage
-----
  # Ingest a PDF
  python cli.py ingest my_paper.pdf

  # Ingest all PDFs in a folder
  python cli.py ingest ./papers/

  # Ask a question (interactive REPL if no question given)
  python cli.py query "What is the main contribution?"

  # Show index statistics
  python cli.py stats

Options
-------
  --db        Path to vector database (default: ./rag_db)
  --llm       Ollama LLM model (default: mistral)
  --embed     Embedding model (default: mxbai-embed-large)
  --reranker  Reranker model (default: qllama/bge-reranker-v2-m3)
  --k         Retrieval candidates (default: 20)
  --top-n     Passages after reranking (default: 5)
  --alpha     Dense/sparse balance 0-1 (default: 0.6)
  --chunk     Chunk size in words (default: 512)
  --overlap   Chunk overlap in words (default: 64)
  --quiet     Suppress verbose pipeline logging
"""

import argparse
import sys
import json
from pathlib import Path

from rag_pipeline import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rag",
        description="RAG pipeline: mxbai-embed-large + BGE reranker + Ollama LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("command", choices=["ingest", "query", "stats", "repl"],
                   help="Command to run")
    p.add_argument("args", nargs="*", help="Command arguments")

    p.add_argument("--db",       default="./rag_db",          help="Vector DB directory")
    p.add_argument("--llm",      default="mistral",             help="Ollama generation model")
    p.add_argument("--embed",    default="mxbai-embed-large",   help="Embedding model")
    p.add_argument("--reranker", default="qllama/bge-reranker-v2-m3",    help="BGE reranker model")
    p.add_argument("--k",        type=int, default=20,         help="Retrieval candidates")
    p.add_argument("--top-n",    type=int, default=5,          help="Passages after reranking")
    p.add_argument("--alpha",    type=float, default=0.6,      help="Dense/sparse balance")
    p.add_argument("--chunk",    type=int, default=512,        help="Chunk size (words)")
    p.add_argument("--overlap",  type=int, default=64,         help="Chunk overlap (words)")
    p.add_argument("--quiet",    action="store_true",          help="Suppress verbose output")
    return p


def make_pipeline(args: argparse.Namespace) -> RAGPipeline:
    return RAGPipeline(
        store_dir=args.db,
        llm_model=args.llm,
        embed_model=args.embed,
        rerank_model=args.reranker,
        retrieval_k=args.k,
        rerank_top_n=args.top_n,
        hybrid_alpha=args.alpha,
        chunk_size=args.chunk,
        chunk_overlap=args.overlap,
    )


def cmd_ingest(rag: RAGPipeline, paths: list[str], quiet: bool) -> None:
    if not paths:
        print("Usage: rag ingest <file.pdf | directory> ...")
        sys.exit(1)

    total = 0
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  ✗ Not found: {path}")
            continue
        if path.is_dir():
            n = rag.ingest_directory(path)
        elif path.suffix.lower() == ".pdf":
            n = rag.ingest_pdf(path)
        else:
            # treat as plain text
            text = path.read_text(encoding="utf-8", errors="replace")
            n = rag.ingest_texts([text], source=path.name)
        total += n

    print(f"\n  ✓ Total chunks ingested: {total}")
    print(f"  ✓ Index size: {rag.store.count} documents")


def cmd_query(rag: RAGPipeline, question: str, quiet: bool) -> None:
    result = rag.query(question, verbose=not quiet)
    if quiet:
        print(result["answer"])
    # timings always printed
    print(
        f"\n  Timings — retrieval: {result['retrieval_time']:.2f}s  "
        f"rerank: {result['rerank_time']:.2f}s  "
        f"generate: {result['generate_time']:.2f}s"
    )


def cmd_repl(rag: RAGPipeline, quiet: bool) -> None:
    print(f"\nRAG REPL — {rag.store.count} docs indexed")
    print("Type your question and press Enter. Ctrl-C or 'quit' to exit.\n")
    while True:
        try:
            q = input(">>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        try:
            cmd_query(rag, q, quiet)
        except Exception as e:
            print(f"  Error: {e}")


def cmd_stats(rag: RAGPipeline) -> None:
    s = rag.stats()
    print(json.dumps(s, indent=2))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rag = make_pipeline(args)

    if args.command == "ingest":
        cmd_ingest(rag, args.args, args.quiet)
    elif args.command == "query":
        if not args.args:
            print("Provide a question: rag query \"your question here\"")
            sys.exit(1)
        cmd_query(rag, " ".join(args.args), args.quiet)
    elif args.command == "repl":
        cmd_repl(rag, args.quiet)
    elif args.command == "stats":
        cmd_stats(rag)


if __name__ == "__main__":
    main()
