#!/usr/bin/env python3
"""Quick test to check for initialization errors."""

import sys
from pathlib import Path

print("Testing imports and initialization...")

try:
    print("1. Importing RAGPipeline...")
    from rag_pipeline import RAGPipeline
    print("   ✓ RAGPipeline imported")

    print("\n2. Creating RAGPipeline instance...")
    rag = RAGPipeline(
        store_dir="./rag_db_test",
        llm_model="mistral",
        embed_model="nomic-embed-text",
        rerank_model="bge-reranker-m3",
        retrieval_k=20,
        rerank_top_n=5,
        hybrid_alpha=0.6,
        chunk_size=1000,
        chunk_overlap=200,
    )
    print("   ✓ RAGPipeline created successfully")

    print("\n3. Checking vector store...")
    print(f"   Store: {rag.store}")
    print(f"   Count: {rag.store.count}")

    print("\n✅ All initialization tests passed!")

except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}")
    print(f"   Message: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
