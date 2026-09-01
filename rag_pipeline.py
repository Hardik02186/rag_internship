"""
rag_pipeline.py
===============
End-to-end RAG pipeline:

  PDF / text  →  PDFLoader  →  chunks (Documents)
                                    │
                              OllamaEmbedder
                            (mxbai-embed-large)
                                    │
                              VectorStore (disk-persisted)
                            dense + BM25 hybrid retrieval
                                    │
                              BGEReranker
                           (bge-reranker-v2-m3)
                                    │
                            LLM (any Ollama model)
                                    │
                               Final Answer
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np

from embedder    import OllamaEmbedder
from vector_store import VectorStore, Document
from reranker    import BGEReranker
from pdf_loader  import PDFLoader, ChunkConfig


OLLAMA_BASE = "http://localhost:11434"

SYSTEM_PROMPT = """\
You are a precise, helpful research assistant. Answer the user's question \
using ONLY the retrieved context passages below. \
Cite passage numbers as [P1], [P2] etc. \
If the context is insufficient, say so clearly — do not hallucinate."""


# ─── LLM generation (Ollama chat) ─────────────────────────────────────────────

def _ollama_chat(
    messages: list[dict],
    model: str,
    system: str | None = None,
    temperature: float = 0.1,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if system:
        payload["system"] = system
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat", data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.loads(r.read())
        return resp["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise ConnectionError(f"Ollama LLM unreachable: {e}") from e


# ─── RAGPipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline with persistent vector store and BGE reranker.

    Parameters
    ----------
    store_dir : str | Path
        Where to persist the vector database (created if absent).
    llm_model : str
        Ollama model for generation (e.g. "llama3", "mistral", "phi4").
    embed_model : str
        Ollama embedding model (default: "mxbai-embed-large").
    rerank_model : str
        Ollama reranker model (default: "qllama/bge-reranker-v2-m3").
    retrieval_k : int
        Number of candidates from hybrid retrieval before reranking.
    rerank_top_n : int
        Passages kept after reranking (passed to the LLM).
    hybrid_alpha : float
        Dense/sparse balance: 1.0 = pure dense, 0.0 = pure BM25.
    chunk_size : int
        Target words per chunk during PDF ingestion.
    chunk_overlap : int
        Word overlap between adjacent chunks.
    embed_batch_size : int
        Texts per embedding API call.
    """

    def __init__(
        self,
        store_dir: str | Path = "./rag_db",
        llm_model: str = "mistral",
        embed_model: str = "mxbai-embed-large",
        rerank_model: str = "qllama/bge-reranker-v2-m3",
        retrieval_k: int = 20,
        rerank_top_n: int = 5,
        hybrid_alpha: float = 0.6,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embed_batch_size: int = 64,
    ) -> None:
        self.llm_model    = llm_model
        self.retrieval_k  = retrieval_k
        self.rerank_top_n = rerank_top_n
        self.hybrid_alpha = hybrid_alpha

        self.embedder = OllamaEmbedder(
            model=embed_model,
            batch_size=embed_batch_size,
        )
        self.store = VectorStore(
            store_dir=store_dir,
            embed_model=embed_model,
        )
        self.reranker = BGEReranker(model=rerank_model)
        self.loader   = PDFLoader(ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ))

    # ── Ingestion ─────────────────────────────────────────────────────────

    def ingest_pdf(self, path: str | Path, save: bool = True) -> int:
        """Ingest a single PDF. Returns number of chunks added."""
        docs = self.loader.load_file(path)
        return self._index_docs(docs, save=save)

    def ingest_pdfs(self, paths: list[str | Path], save: bool = True) -> int:
        """Ingest multiple PDFs."""
        all_docs = []
        for p in paths:
            docs = self.loader.load_file(p)
            all_docs.extend(docs)
        return self._index_docs(all_docs, save=save)

    def ingest_directory(self, directory: str | Path, save: bool = True) -> int:
        """
        Ingest all PDFs in a directory.
        Also removes embeddings of PDFs that have been deleted.
        """
        directory = Path(directory)
        
        # Find current PDF files
        pdfs = sorted(directory.glob("*.pdf"))
        pdf_names = {pdf.name for pdf in pdfs}
        
        # Clean up embeddings from deleted PDFs
        self._cleanup_deleted_pdfs(pdf_names)
        
        if not pdfs:
            print(f"  No PDFs found in {directory}")
            return 0
        
        print(f"  Found {len(pdfs)} PDF(s) in {directory}")
        return self.ingest_pdfs(pdfs, save=save)

    def _cleanup_deleted_pdfs(self, current_pdf_names: set[str]) -> None:
        """Remove embeddings of PDFs that no longer exist in the directory."""
        # Get all unique sources currently in the vector store
        all_sources = set()
        try:
            results = self.store.collection.get()
            if results and results["metadatas"]:
                for meta in results["metadatas"]:
                    source = meta.get("source", "")
                    if source and source != "manual":  # Skip non-PDF sources
                        all_sources.add(source)
        except:
            return
        
        # Find sources that are no longer in the directory
        deleted_sources = all_sources - current_pdf_names
        
        # Remove them from the vector store
        if deleted_sources:
            print(f"\n  🗑️  Cleaning up {len(deleted_sources)} deleted PDF(s):")
            for source in sorted(deleted_sources):
                self.store.delete_by_source(source)


    def ingest_texts(
        self,
        texts: list[str],
        source: str = "manual",
        save: bool = True,
    ) -> int:
        """Ingest plain text strings directly (no chunking applied)."""
        import uuid
        docs = [
            Document(
                id=str(uuid.uuid4()),
                text=t,
                source=source,
                chunk_index=i,
            )
            for i, t in enumerate(texts)
        ]
        return self._index_docs(docs, save=save)

    def _index_docs(self, docs: list[Document], save: bool = True) -> int:
        if not docs:
            print("  No documents to index.")
            return 0

        print(f"\n  Embedding {len(docs)} chunks…")
        t0 = time.time()
        embeddings = self.embedder.embed_documents([d.text for d in docs])
        dt = time.time() - t0
        print(f"  Embedded in {dt:.1f}s ({dt/len(docs)*1000:.0f}ms/chunk)")

        self.store.add(docs, embeddings, auto_save=save)
        print(f"  Index total: {self.store.count} documents")
        return len(docs)

    # ── Query ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        verbose: bool = True,
        temperature: float = 0.1,
    ) -> dict:
        """
        Full pipeline: embed → hybrid retrieve → BGE rerank → generate.

        Returns
        -------
        dict with keys:
          answer         : str — the LLM-generated answer
          sources        : list[Document] — top-N passages used
          rerank_scores  : list[float]
          retrieval_time : float (seconds)
          rerank_time    : float (seconds)
          generate_time  : float (seconds)
        """
        if self.store.count == 0:
            raise RuntimeError("No documents indexed. Call ingest_* first.")

        sep = "─" * 60
        if verbose:
            print(f"\n{sep}\nQuery: {question}\n{sep}")

        # 1. Embed query
        t0 = time.time()
        q_vec = self.embedder.embed_query(question)
        if verbose:
            print(f"\n[1] Query embedded ({(time.time()-t0)*1000:.0f}ms)")

        # 2. Hybrid retrieval
        t0 = time.time()
        candidates = self.store.hybrid_search(
            q_vec, question,
            top_k=self.retrieval_k,
            alpha=self.hybrid_alpha,
        )
        retrieval_time = time.time() - t0
        if verbose:
            print(f"\n[2] Hybrid retrieval → {len(candidates)} candidates ({retrieval_time*1000:.0f}ms)")
            for i, (doc, score) in enumerate(candidates[:5]):
                print(f"    [{i+1}] {score:.3f}  [{doc.source}]  {doc.text[:80]}…")
            if len(candidates) > 5:
                print(f"    … +{len(candidates)-5} more")

        # 3. BGE Rerank
        t0 = time.time()
        reranked = self.reranker.rerank(question, candidates, self.rerank_top_n)
        rerank_time = time.time() - t0
        if verbose:
            print(f"\n[3] BGE Reranked → top {len(reranked)} ({rerank_time*1000:.0f}ms)")
            for i, (doc, score) in enumerate(reranked):
                print(f"    [P{i+1}] score={score:.4f}  [{doc.source} chunk={doc.chunk_index}]  {doc.text[:80]}…")

        # 4. Build context & generate
        context_parts = []
        for i, (doc, score) in enumerate(reranked):
            meta = f"[source: {doc.source}, chunk: {doc.chunk_index}]"
            context_parts.append(f"[P{i+1}] {meta}\n{doc.text}")
        context = "\n\n".join(context_parts)

        prompt = f"Context Passages:\n\n{context}\n\nQuestion: {question}"
        if verbose:
            print(f"\n[4] Generating with {self.llm_model}…")
        t0 = time.time()
        answer = _ollama_chat(
            [{"role": "user", "content": prompt}],
            model=self.llm_model,
            system=SYSTEM_PROMPT,
            temperature=temperature,
        )
        generate_time = time.time() - t0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Answer ({generate_time:.1f}s):\n{answer}")
            print(f"{'='*60}\n")

        return {
            "answer": answer,
            "sources": [doc for doc, _ in reranked],
            "rerank_scores": [score for _, score in reranked],
            "retrieval_time": retrieval_time,
            "rerank_time": rerank_time,
            "generate_time": generate_time,
        }

    # ── Utilities ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return index statistics."""
        return {
            "total_documents": self.store.count,
            "embed_model": self.embedder.model,
            "rerank_model": self.reranker.model,
            "llm_model": self.llm_model,
            "store_dir": str(self.store.store_dir),
            "hybrid_alpha": self.hybrid_alpha,
            "retrieval_k": self.retrieval_k,
            "rerank_top_n": self.rerank_top_n,
        }
