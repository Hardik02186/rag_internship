from __future__ import annotations

import json
import math
import time
import urllib.request
import urllib.error

import numpy as np

from vector_store import Document


RERANKER_MODEL_DEFAULT = "qllama/bge-reranker-v2-m3"
OLLAMA_BASE = "http://localhost:11434"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class BGEReranker:

    def __init__(
        self,
        model: str = RERANKER_MODEL_DEFAULT,
        base_url: str = OLLAMA_BASE,
        batch_size: int = 32,
    ) :
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self._has_native_rerank: bool | None = None  # discovered lazily

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Document, float]],
        top_n: int = 5,
    ) -> list[tuple[Document, float]]:
        
        if not candidates:
            return []

        docs = [doc for doc, _ in candidates]
        passages = [doc.text for doc in docs]

        print(f"  BGE Reranking {len(passages)} candidates → top {top_n}…")

        if self._has_native_rerank is None:
            self._has_native_rerank = self._probe_native_rerank()

        if self._has_native_rerank:
            scores = self._native_rerank(query, passages)
        else:
            scores = self._embed_rerank(query, passages)

        paired = list(zip(docs, scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:top_n]

    # ── Ollama native /api/rerank (≥ 0.3.0) ─────────────────────────────

    def _probe_native_rerank(self) -> bool:
        """Check if /api/rerank exists."""
        try:
            payload = json.dumps({
                "model": self.model,
                "query": "test",
                "documents": ["test doc"],
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/rerank",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
            return True
        except Exception:
            return False

    def _native_rerank(self, query: str, passages: list[str]) -> list[float]:
        """Use Ollama's dedicated /api/rerank endpoint."""
        payload = json.dumps({
            "model": self.model,
            "query": query,
            "documents": passages,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/rerank",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        # Response: {"results": [{"index": 0, "relevance_score": 0.97}, ...]}
        results = resp.get("results", [])
        scores = [0.0] * len(passages)
        for r in results:
            scores[r["index"]] = float(r["relevance_score"])
        return scores

    # ── Embedding-based fallback ──────────────────────────────────────────

    def _embed_rerank(self, query: str, passages: list[str]) -> list[float]:
        """
        Fallback: feed (query, passage) pairs through the reranker as an
        embedding model and read the first logit dimension as the score.
        BGE reranker models output a single relevance logit when used this way.
        """
        scores: list[float] = []
        for i in range(0, len(passages), self.batch_size):
            batch = passages[i : i + self.batch_size]
            # BGE cross-encoder input format: <query>[SEP]<passage>
            inputs = [f"{query}\n\n{p[:1500]}" for p in batch]
            raw_scores = self._embed_batch(inputs)
            scores.extend(raw_scores)
        return scores

    def _embed_batch(self, texts: list[str]) -> list[float]:
        payload = json.dumps({"model": self.model, "input": texts}).encode()
        url = f"{self.base_url}/api/embed"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.loads(r.read())
                embeddings = resp.get("embeddings", [])
                # First dimension is the raw relevance logit for cross-encoders
                return [_sigmoid(float(e[0])) if e else 0.0 for e in embeddings]
            except urllib.error.URLError as e:
                if attempt == 2:
                    raise ConnectionError(
                        f"Ollama reranker unreachable. "
                        f"Run: ollama pull {self.model}\n  {e}"
                    ) from e
                time.sleep(1.5)
        return [0.0] * len(texts)
