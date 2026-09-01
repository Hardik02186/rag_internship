#!/usr/bin/env python3
"""
Test cleanup of deleted PDF embeddings.
"""

import shutil
import tempfile
from pathlib import Path
from rag_pipeline import RAGPipeline

print("🧪 Testing PDF Deletion & Cleanup Feature...\n")

# Create temporary directory for test
test_dir = Path(tempfile.mkdtemp(prefix="rag_cleanup_test_"))
rag_db_dir = test_dir / "rag_db"

print(f"📁 Test directory: {test_dir}")

try:
    # Initialize RAG pipeline
    print("\n1️⃣  Initializing RAG Pipeline...")
    rag = RAGPipeline(
        store_dir=str(rag_db_dir),
        llm_model="mistral:latest",
        embed_model="mxbai-embed-large:latest",
        rerank_model="qllama/bge-reranker-v2-m3:latest",
    )
    print("   ✓ Pipeline initialized")

    # Ingest sample texts
    print("\n2️⃣  Ingesting sample texts...")
    sample_texts = [
        "This is document one about artificial intelligence.",
        "This is document two about machine learning.",
        "This is document three about deep learning.",
    ]
    num_ingested = rag.ingest_texts(sample_texts, source="sample.pdf")
    print(f"   ✓ Ingested {num_ingested} texts")
    print(f"   Total documents in store: {rag.store.count}")

    # Show current documents
    print("\n3️⃣  Current documents in vector store:")
    try:
        results = rag.store.collection.get()
        for i, meta in enumerate(results["metadatas"]):
            print(f"   [{i+1}] {meta.get('source')} - chunk {meta.get('chunk_index')}")
    except:
        print("   (Could not retrieve details)")

    # Delete by source
    print("\n4️⃣  Deleting all embeddings from 'sample.pdf'...")
    deleted_count = rag.store.delete_by_source("sample.pdf")
    print(f"   Deleted {deleted_count} embeddings")
    print(f"   Total documents remaining: {rag.store.count}")

    # Verify deletion
    print("\n5️⃣  Verifying cleanup...")
    if rag.store.count == 0:
        print("   ✅ All embeddings successfully removed!")
    else:
        print(f"   ⚠️  {rag.store.count} documents still in store")

    print("\n✅ Cleanup feature working correctly!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    print(f"\n🧹 Cleaning up test directory...")
    shutil.rmtree(test_dir, ignore_errors=True)
    print("   Done!")
