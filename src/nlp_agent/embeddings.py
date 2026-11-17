"""Modelli di embedding: sentence-transformers (reale) e hashing (test/offline)."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Modelli multilingua, es. intfloat/multilingual-e5-small (gli e5 vogliono i prefissi query/passage)."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.e5 = "e5" in model_name.lower()

    def _enc(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_documents(self, texts):
        return self._enc([f"passage: {t}" for t in texts] if self.e5 else texts)

    def embed_query(self, text):
        return self._enc([f"query: {text}" if self.e5 else text])[0]


class HashEmbedder:
    """Bag-of-words con hashing: deterministico, senza download. Solo per test e prove offline."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim)
        for tok in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def get_embedder(name: str) -> Embedder:
    return HashEmbedder() if name == "hash" else SentenceTransformerEmbedder(name)
