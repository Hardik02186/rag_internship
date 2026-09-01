"""
example.py
==========
Quick demo of the RAG pipeline.
Run this after:
  ollama serve
  ollama pull nomic-embed-text
  ollama pull bge-reranker-m3
  ollama pull mistral
"""

from rag_pipeline import RAGPipeline

# ── Create pipeline ───────────────────────────────────────────────────────────
rag = RAGPipeline(
    store_dir="./rag_db",          # persisted here — survives restarts
    llm_model="mistral",
    embed_model="nomic-embed-text",
    rerank_model="bge-reranker-m3",
    retrieval_k=20,                # pull 20 candidates from hybrid search
    rerank_top_n=5,                # BGE reranker keeps best 5
    hybrid_alpha=0.6,              # 60% dense, 40% BM25
    chunk_size=600,               # characters per chunk (LangChain RecursiveCharacterTextSplitter)
    chunk_overlap=200,             # character overlap between chunks
)

# ── Ingest PDFs ───────────────────────────────────────────────────────────────
# Single PDF:
# rag.ingest_pdf("my_paper.pdf")

# Multiple PDFs:
# rag.ingest_pdfs(["paper1.pdf", "paper2.pdf", "manual.pdf"])

# Entire directory:
# rag.ingest_directory("./papers/")

# Or plain text (for testing):
rag.ingest_texts([
    "Retrieval-Augmented Generation (RAG) combines a retrieval system with "
    "a language model. The retriever fetches relevant passages from a knowledge "
    "base; the generator uses them as grounded context to produce accurate answers. "
    "RAG reduces hallucination by anchoring generation to retrieved evidence.",

    "nomic-embed-text v1.5 is an open-source embedding model trained with "
    "Matryoshka Representation Learning (MRL). It supports up to 8192 input tokens "
    "and achieves state-of-the-art performance on the MTEB benchmark. "
    "It uses asymmetric prompting: 'search_query:' for queries and "
    "'search_document:' for passages.",

    "BGE-Reranker-v2-M3 is a cross-encoder reranker from BAAI. Unlike bi-encoders "
    "that embed query and passage independently, cross-encoders attend over the "
    "concatenated (query, passage) pair, producing a single relevance score. "
    "This makes them more accurate but slower than bi-encoder retrieval.",

    "Hybrid search combines dense vector retrieval (semantic) with sparse BM25 "
    "(keyword) retrieval. Reciprocal Rank Fusion (RRF) or weighted score fusion "
    "merges the two ranked lists. Hybrid search outperforms either approach alone, "
    "especially on queries with rare or technical keywords.",
], source="demo")

# ── Print stats ───────────────────────────────────────────────────────────────
import json
print("\nIndex stats:")
print(json.dumps(rag.stats(), indent=2))

# ── Query ─────────────────────────────────────────────────────────────────────
query_text = input("Enter your query: ")
result = rag.query(query_text)

# Access result components:
print("\nSources used:")
for i, doc in enumerate(result["sources"]):
    print(f"  [P{i+1}] {doc.source} chunk={doc.chunk_index} "
          f"score={result['rerank_scores'][i]:.4f}")
