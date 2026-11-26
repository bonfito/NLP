"""Vector store su Chroma (persistente, locale). Gli embeddings sono calcolati fuori da Chroma."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

from .embeddings import Embedder
from .ingest.chunking import Chunk


class Passage(TypedDict):
    id: str
    text: str
    file: str
    part: str
    macro: str
    kind: str
    locator: str
    stance: str
    score: float


COLLECTION = "nlp_course"
RRF_K = 60   # costante della Reciprocal Rank Fusion
_TOKEN = re.compile(r"[A-Za-z0-9À-ÿ]+")


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN.findall(text) if len(w) > 1]


class BM25:
    """BM25 minimale sui testi dei chunk: trova i termini esatti (sigle, nomi, numeri) che gli embeddings diluiscono."""

    def __init__(self, ids: list[str], docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.ids, self.k1, self.b = ids, k1, b
        self.tf = [Counter(_tokens(d)) for d in docs]
        self.len = [sum(c.values()) for c in self.tf]
        self.avg = (sum(self.len) / len(self.len)) if self.len else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1) for t, f in df.items()}

    def top(self, query: str, n: int) -> list[str]:
        q = set(_tokens(query))
        scored = []
        for i, c in enumerate(self.tf):
            sc = 0.0
            for t in q:
                f = c.get(t)
                if f:
                    sc += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if sc > 0:
                scored.append((sc, i))
        scored.sort(reverse=True)
        return [self.ids[i] for _, i in scored[:n]]


class VectorStore:
    def __init__(self, index_dir: str | Path, embedder: Embedder):
        import chromadb

        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(index_dir))
        self.col = self._collection()
        self._lex: tuple[BM25, dict] | None = None

    def _collection(self):
        return self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"},
                                                    embedding_function=None)

    def reset(self) -> None:
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        self.col = self._collection()
        self._lex = None

    def count(self) -> int:
        return self.col.count()

    def add(self, chunks: list[Chunk], batch: int = 64) -> None:
        self._lex = None
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            self.col.upsert(ids=[c.id for c in part], documents=[c.text for c in part],
                            embeddings=self.embedder.embed_documents([c.text for c in part]),
                            metadatas=[c.metadata for c in part])

    def _lexical(self) -> tuple[BM25, dict]:
        """Indice BM25 costruito (e tenuto in memoria) dai testi già salvati in Chroma: non serve reindicizzare."""
        if self._lex is None:
            res = self.col.get(include=["documents", "metadatas"])
            info = {pid: (doc, meta) for pid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"])}
            self._lex = (BM25(res["ids"], res["documents"]), info)
        return self._lex

    @staticmethod
    def _passage(pid: str, doc: str, meta: dict, score: float) -> Passage:
        return Passage(id=pid, text=doc, file=meta["file"], part=meta["part"], macro=meta["macro"],
                       kind=meta["kind"], locator=meta.get("locator", ""),
                       stance=meta.get("stance", ""), score=score)

    def query(self, text: str, k: int = 6, where: dict | None = None, hybrid: bool = True) -> list[Passage]:
        """Ricerca ibrida: classifica per similarità (embeddings) e per parole esatte (BM25), poi fonde le due
        classifiche con Reciprocal Rank Fusion. Con `where` o hybrid=False usa solo gli embeddings."""
        if self.count() == 0:
            return []
        res = self.col.query(query_embeddings=[self.embedder.embed_query(text)],
                             n_results=min(k, self.count()), where=where)
        dense = [self._passage(pid, doc, meta, round(1 - dist, 4)) for pid, doc, meta, dist in
                 zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])]
        if not hybrid or where:
            return dense
        bm25, info = self._lexical()
        fused: dict[str, float] = {}
        for rank, p in enumerate(dense):
            fused[p["id"]] = fused.get(p["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, pid in enumerate(bm25.top(text, k)):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        by_id = {p["id"]: p for p in dense}
        out = []
        for pid in sorted(fused, key=fused.get, reverse=True)[:k]:
            out.append(by_id[pid] if pid in by_id else self._passage(pid, *info[pid], score=0.0))
        return out
