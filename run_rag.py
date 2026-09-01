import sys
from pathlib import Path
from rag_pipeline import RAGPipeline


def main():
    # Get PDF folder from command line or default to current directory
    if len(sys.argv) > 1:
        pdf_folder = sys.argv[1]
    else:
        pdf_folder = "."

    pdf_folder = Path(pdf_folder)
    if not pdf_folder.exists():
        print(f"Folder not found: {pdf_folder}")
        return

    # ── Initialize RAG Pipeline ───────────────────────────────────────────
    print("Initializing RAG Pipeline…")
    rag = RAGPipeline(
        store_dir="./rag_db",
        llm_model="mistral:latest",
        embed_model="mxbai-embed-large:latest",
        rerank_model="qllama/bge-reranker-v2-m3:latest",
        retrieval_k=10,
        rerank_top_n=5,
        hybrid_alpha=0.6,
        chunk_size=600,
        chunk_overlap=200,
    )

    # ── Ingest PDFs from folder ──────────────────────────────────────────
    print(f"\n Looking for PDFs in: {pdf_folder.resolve()}")
    num_docs = rag.ingest_directory(pdf_folder, save=True)

    if num_docs == 0:
        print(" No PDFs found or no text could be extracted.")
        return

    print(f"\n Successfully indexed {num_docs} document chunks")
    print(json.dumps(rag.stats(), indent=2))

    # ── Interactive Query Loop ───────────────────────────────────────────
    print("\n" + "="*60)
    print(" RAG Pipeline Ready!")
    print("="*60)
    print("\nType your questions below. Press Ctrl+C to exit.\n")

    try:
        while True:
            question = input(" Your question: ").strip()
            if not question:
                continue

            result = rag.query(question, verbose=True)

            # Print sources
            print("\n Sources used:")
            for i, doc in enumerate(result["sources"]):
                score = result["rerank_scores"][i]
                print(f"  [P{i+1}] score={score:.4f} | {doc.source} (chunk {doc.chunk_index})")
                print(f"        {doc.text[:100]}…\n")

    except KeyboardInterrupt:
        print("\n\nGoodbye!")


if __name__ == "__main__":
    import json
    main()
