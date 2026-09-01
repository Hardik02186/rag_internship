"""
embedder.py
===========
Proper embedding via Ollama's /api/embed endpoint (batch-native).
Model: nomic-embed-text (nomic-embed-text-v1.5, 768-dim, 8192 ctx)

Ollama's /api/embed accepts a list of strings in one call — we use this
to avoid N round-trips for large ingestion jobs.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Sequence

import numpy as np


EMBED_MODEL_DEFAULT = "mxbai-embed-large"
OLLAMA_BASE = "http://localhost:11434"


class OllamaEmbedder:
    """
    Batch embedder backed by Ollama's /api/embed endpoint.

    Parameters
    ----------
    model : str
        Ollama model tag, e.g. "nomic-embed-text".
    base_url : str
        Ollama server base URL.
    batch_size : int
        Max texts per HTTP request. Ollama handles batches natively.
    retry : int
        Number of retries on transient errors.
    """

    def __init__(
        self,
        model: str = EMBED_MODEL_DEFAULT,
        base_url: str = OLLAMA_BASE,
        batch_size: int = 64,
        retry: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.retry = retry
        self._dim: int | None = None

    # ── public ────────────────────────────────────────────────────────────

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed a list of strings.

        Returns
        -------
        np.ndarray  shape (N, D), dtype float32, L2-normalised
        """
        if not texts:
            raise ValueError("texts must be non-empty")

        texts = [t.strip() or " " for t in texts]   # Ollama rejects empty strings
        all_vecs: list[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vecs = self._embed_batch(batch)
            all_vecs.append(vecs)

        mat = np.vstack(all_vecs).astype(np.float32)
        return self._l2_norm(mat)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns shape (D,)."""
        # nomic-embed-text uses a search_query prefix for asymmetric retrieval
        prefixed = f"search_query: {text}"
        return self.embed([prefixed])[0]

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed passage/document strings. Returns shape (N, D)."""
        # nomic-embed-text uses search_document prefix for passages
        prefixed = [f"search_document: {t}" for t in texts]
        return self.embed(prefixed)

    @property
    def dim(self) -> int:
        if self._dim is None:
            v = self.embed_query("warmup")
            self._dim = v.shape[0]
        return self._dim

    # ── internal ──────────────────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        # Try modern batch endpoint first (/api/embed, Ollama >= 0.1.26),
        # fall back to legacy single-text endpoint (/api/embeddings).
        try:
            return self._embed_batch_modern(texts)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return self._embed_batch_legacy(texts)
            raise

    def _embed_batch_modern(self, texts: list[str]) -> np.ndarray:
        """POST /api/embed — Ollama >= 0.1.26, accepts list input."""
        payload = json.dumps({"model": self.model, "input": texts}).encode()
        url = f"{self.base_url}/api/embed"
        for attempt in range(self.retry):
            try:
                req = urllib.request.Request(
                    url, data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=500) as r:
                    resp = json.loads(r.read())
                # {"embeddings": [[...], [...]]}
                return np.array(resp["embeddings"], dtype=np.float32)
            except urllib.error.HTTPError:
                raise   # let caller decide (404 → fallback)
            except urllib.error.URLError as e:
                if attempt == self.retry - 1:
                    raise ConnectionError(
                        f"Ollama unreachable. Is 'ollama serve' running?\n  {e}"
                    ) from e
                time.sleep(1.5 * (attempt + 1))

    def _embed_batch_legacy(self, texts: list[str]) -> np.ndarray:
        """POST /api/embeddings — older Ollama, one request per text."""
        url = f"{self.base_url}/api/embeddings"
        vecs = []
        for text in texts:
            payload = json.dumps({"model": self.model, "prompt": text}).encode()
            for attempt in range(self.retry):
                try:
                    req = urllib.request.Request(
                        url, data=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(req, timeout=500) as r:
                        resp = json.loads(r.read())
                    vecs.append(resp["embedding"])
                    break
                except urllib.error.URLError as e:
                    if attempt == self.retry - 1:
                        raise ConnectionError(
                            f"Ollama unreachable at {url}.\n"
                            f"Run: ollama serve && ollama pull {self.model}\n  {e}"
                        ) from e
                    time.sleep(1.5 * (attempt + 1))
        return np.array(vecs, dtype=np.float32)

    @staticmethod
    def _l2_norm(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-9)