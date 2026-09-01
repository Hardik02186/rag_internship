# RAG System v2 — nomic-embed-text + BGE Reranker + Persistent VectorDB

Production-grade RAG pipeline for large PDF corpora.

| Component | Implementation |
|-----------|---------------|
| Embeddings | `nomic-embed-text` via Ollama `/api/embed` (batch, asymmetric prefixes) |
| Vector DB | Persistent `.npz` + JSON on disk (numpy + scipy cKDTree) |
| Sparse search | BM25 — pure Python, rebuilt on load |
| Fusion | Weighted score fusion (dense + BM25) |
| Reranker | `bge-reranker-m3` via Ollama `/api/rerank` (native) or `/api/embed` fallback |
| PDF parsing | `pypdf` with cross-page chunking + metadata |
| Generation | Any Ollama LLM |

---

## Setup

```bash
# 1. Start Ollama
ollama serve

# 2. Pull models
ollama pull nomic-embed-text    # 768-dim, 8192 ctx, MTEB SOTA
ollama pull bge-reranker-m3     # cross-encoder reranker
ollama pull llama3              # or: mistral, phi4, gemma2, qwen2.5

# 3. Python deps (stdlib + numpy + scipy + pypdf — all pre-installed)
pip install numpy scipy pypdf
```

---

## Quick Start

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline(
    store_dir="./rag_db",       # persisted here
    llm_model="mistral",
    embed_model="nomic-embed-text",
    rerank_model="bge-reranker-m3",
    retrieval_k=20,             # hybrid retrieval candidates
    rerank_top_n=5,             # kept after BGE reranking
    hybrid_alpha=0.6,           # 0=BM25 only, 1=dense only
    chunk_size=512,             # words per chunk
    chunk_overlap=64,
)

# Ingest
rag.ingest_pdf("paper.pdf")
rag.ingest_directory("./papers/")   # all PDFs in a folder
rag.ingest_pdfs(["a.pdf", "b.pdf"])

# Query
result = rag.query("What is the main contribution?")
print(result["answer"])
print(result["sources"])        # list[Document]
print(result["rerank_scores"])  # list[float]
```

---

## CLI

```bash
# Ingest
python cli.py ingest paper.pdf
python cli.py ingest ./papers/
python cli.py ingest --db ./my_db --chunk 400 paper.pdf

# Query (single shot)
python cli.py query "What are the limitations?"
python cli.py query --quiet "summarise the methods"

# Interactive REPL
python cli.py repl

# Stats
python cli.py stats --db ./rag_db
```

---

## Architecture & Data Flow

```
PDFs / text
    │
    └─► PDFLoader
          - pypdf text extraction
          - cross-page chunking (512 words, 64 overlap)
          - metadata: source, page, chunk_index
          │
          ▼
    OllamaEmbedder (nomic-embed-text)
          - asymmetric: "search_document:" prefix for passages
          - batch API: /api/embed with list input
          - L2-normalised float32 output
          │
          ▼
    VectorStore (disk-persisted)
          vectors.npz  ← float32 matrix (N × 768)
          docs.json    ← text + metadata
          meta.json    ← model, dim, count, timestamps
          │
    ┌─────┴──────┐
    │            │
  Dense        BM25
 (cosine)   (Okapi BM25)
    │            │
    └─────┬──────┘
          │  Weighted Score Fusion (alpha=0.6)
          │  top-K candidates
          ▼
    BGEReranker (bge-reranker-m3)
          - /api/rerank endpoint (Ollama ≥ 0.3)
          - fallback: /api/embed cross-encoder scoring
          │  top-N passages
          ▼
    Ollama LLM (llama3 / mistral / …)
          - system prompt anchors to retrieved context
          - passage citations [P1][P2]…
          │
          ▼
       Answer
```

---

## Disk Layout

```
rag_db/
  vectors.npz   — compressed numpy array (N × D embeddings)
  docs.json     — document texts + metadata
  meta.json     — model info, dimensions, timestamps
```

The index is **append-only by default** — calling `ingest_*` multiple times grows the store incrementally, each time saving to disk.

---

## Tuning Guide

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `chunk_size` | 512 | Larger (700–900) for narrative text; smaller (200–300) for technical/tables |
| `chunk_overlap` | 64 | Increase to 128 if answers are cut across chunk boundaries |
| `retrieval_k` | 20 | Increase for very large corpora (> 100k chunks) |
| `rerank_top_n` | 5 | Increase to 8–10 for complex multi-hop questions |
| `hybrid_alpha` | 0.6 | Lower (0.4) for keyword-heavy queries; higher (0.8) for semantic |

---

## Files

```
embedder.py      — OllamaEmbedder (nomic-embed-text, batched, asymmetric)
vector_store.py  — VectorStore (.npz persistence), BM25, hybrid search
reranker.py      — BGEReranker (/api/rerank + embed fallback)
pdf_loader.py    — PDFLoader (pypdf, smart chunking, metadata)
rag_pipeline.py  — RAGPipeline (orchestration + LLM generation)
cli.py           — CLI: ingest / query / repl / stats
example.py       — Runnable demo
README.md        — This file
```
