
from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import chromadb


# ─── Document ─────────────────────────────────────────────────────────────────

@dataclass
class Document:
    id: str
    text: str
    source: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        return cls(**d)


# ─── BM25 ─────────────────────────────────────────────────────────────────────

def _tok(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Okapi BM25 — pure Python, zero extra dependencies."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def build(self, texts: list[str]) -> None:
        self._corpus = [_tok(t) for t in texts]
        n = len(self._corpus)
        lengths = [len(d) for d in self._corpus]
        self._avgdl = sum(lengths) / max(n, 1)
        from collections import Counter
        df: Counter = Counter()
        for doc in self._corpus:
            for t in set(doc):
                df[t] += 1
        self._idf = {
            t: math.log((n - cnt + 0.5) / (cnt + 0.5) + 1)
            for t, cnt in df.items()
        }

    def scores(self, query: str) -> np.ndarray:
        qtoks = _tok(query)
        n = len(self._corpus)
        s = np.zeros(n, dtype=np.float32)
        for t in qtoks:
            idf = self._idf.get(t, 0.0)
            if idf == 0:
                continue
            for i, doc in enumerate(self._corpus):
                tf = doc.count(t)
                if tf == 0:
                    continue
                dl = len(doc)
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                s[i] += idf * tf * (self.k1 + 1) / denom
        return s


# ─── VectorStore (Chroma-based) ───────────────────────────────────────────────

class VectorStore:
    """
    Vector store using Chroma for persistence and dense search.
    Hybrid retrieval combines Chroma's cosine similarity with BM25 keyword search.
    """

    def __init__(self, store_dir: str | Path, embed_model: str = "mxbai-embed-large") -> None:
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embed_model = embed_model

        # Initialize Chroma persistent client (new API)
        self.client = chromadb.PersistentClient(
            path=str(self.store_dir / "chroma")
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

        # Document lookup (id → Document object)
        self._doc_map: dict[str, Document] = {}
        self._bm25 = BM25Index()
        
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing documents from Chroma collection."""
        try:
            results = self.collection.get()
            if results and results["ids"]:
                print(f"  ✓ Loaded existing Chroma collection: {len(results['ids'])} documents")
                # Rebuild doc_map from metadata
                for doc_id, meta in zip(results["ids"], results["metadatas"]):
                    doc = Document(
                        id=doc_id,
                        text=meta.get("text", ""),
                        source=meta.get("source", ""),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        metadata=json.loads(meta.get("metadata_json", "{}")),
                    )
                    self._doc_map[doc_id] = doc
                self._rebuild_bm25()
        except Exception as e:
            print(f"  ⚠ No existing collection or error loading: {e}")

    def add(
        self,
        docs: list[Document],
        embeddings: np.ndarray,  # (N, D) float32
        auto_save: bool = True,
    ) -> None:
        """Add documents and embeddings to Chroma."""
        assert len(docs) == len(embeddings), "docs / embeddings length mismatch"

        # Prepare metadata for Chroma
        ids = [d.id for d in docs]
        metadatas = []
        documents = []
        
        for doc in docs:
            self._doc_map[doc.id] = doc
            metadatas.append({
                "source": doc.source,
                "chunk_index": str(doc.chunk_index),
                "text": doc.text,
                "metadata_json": json.dumps(doc.metadata),
            })
            documents.append(doc.text)

        # Add to Chroma collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),  # Convert numpy array to list
            metadatas=metadatas,
            documents=documents,
        )

        # Rebuild BM25 index
        self._rebuild_bm25()

        if auto_save:
            self.save()

    def save(self) -> None:
        """Persist Chroma collection to disk (automatic in new API)."""
        try:
            # In the new Chroma API, data is persisted automatically
            # This is a no-op but kept for backwards compatibility
            print(f"  ✓ Chroma collection persisted to {self.store_dir / 'chroma'}")
        except Exception as e:
            print(f"  ✗ Error persisting: {e}")

    def hybrid_search(
        self,
        query_vec: np.ndarray,
        query_text: str,
        top_k: int = 20,
        alpha: float = 0.6,
    ) -> list[tuple[Document, float]]:
        """
        Hybrid retrieval: weighted fusion of dense (Chroma cosine) and sparse (BM25).
        alpha=1.0 → dense only; alpha=0.0 → BM25 only.
        """
        # Dense search via Chroma
        dense_results = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
        )
        
        dense_scores = {}
        if dense_results and dense_results["distances"] and len(dense_results["distances"]) > 0:
            # Chroma returns distances, convert to similarity (1 - distance for cosine)
            for doc_id, distance in zip(
                dense_results["ids"][0],
                dense_results["distances"][0]
            ):
                # For cosine distance, similarity = 1 - distance
                similarity = 1.0 - distance if distance <= 1.0 else 0.0
                dense_scores[doc_id] = similarity

        # Sparse search via BM25
        sparse_scores_list = self.bm25_search(query_text, top_k)
        sparse_scores = {doc_id: score for doc_id, score in sparse_scores_list}

        # Normalize and fuse
        def norm(scores_dict: dict) -> dict:
            if not scores_dict:
                return {}
            mx = max(scores_dict.values()) or 1e-9
            return {k: v / mx for k, v in scores_dict.items()}

        dense_norm = norm(dense_scores)
        sparse_norm = norm(sparse_scores)

        all_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
        fused = {
            doc_id: alpha * dense_norm.get(doc_id, 0.0) + 
                    (1 - alpha) * sparse_norm.get(doc_id, 0.0)
            for doc_id in all_ids
        }

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._doc_map[doc_id], score) for doc_id, score in ranked]

    def bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """BM25 search returning (doc_id, score) tuples."""
        if not self._doc_map:
            return []
        
        # Create a list of doc IDs in order
        doc_ids = list(self._doc_map.keys())
        scores = self._bm25.scores(query)
        
        k = min(top_k, len(scores))
        if k == 0:
            return []
        
        idxs = np.argpartition(scores, -k)[-k:]
        idxs = idxs[np.argsort(scores[idxs])[::-1]]
        
        return [(doc_ids[int(i)], float(scores[int(i)])) for i in idxs]

    @property
    def count(self) -> int:
        """Total number of documents in the collection."""
        try:
            return self.collection.count()
        except:
            return len(self._doc_map)

    def __repr__(self) -> str:
        return f"VectorStore(docs={self.count}, backend=chroma, dir={self.store_dir})"

    def delete_by_source(self, source: str) -> int:
        """
        Delete all documents from a specific source (e.g., filename).
        
        Returns number of documents deleted.
        """
        doc_ids_to_delete = [
            doc_id for doc_id, doc in self._doc_map.items()
            if doc.source == source
        ]
        
        if not doc_ids_to_delete:
            return 0
        
        # Remove from Chroma collection
        self.collection.delete(ids=doc_ids_to_delete)
        
        # Remove from doc_map
        for doc_id in doc_ids_to_delete:
            del self._doc_map[doc_id]
        
        # Rebuild BM25 index
        self._rebuild_bm25()
        
        print(f"  ✓ Deleted {len(doc_ids_to_delete)} embeddings from source: {source}")
        return len(doc_ids_to_delete)

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from stored documents."""
        texts = [self._doc_map[doc_id].text for doc_id in sorted(self._doc_map.keys())]
        self._bm25.build(texts)

