"""Vector store su Chroma (persistente, locale). Gli embeddings sono calcolati fuori da Chroma."""
from __future__ import annotations

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


class VectorStore:
    def __init__(self, index_dir: str | Path, embedder: Embedder):
        import chromadb

        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(index_dir))
        self.col = self._collection()

    def _collection(self):
        return self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"},
                                                    embedding_function=None)

    def reset(self) -> None:
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        self.col = self._collection()

    def count(self) -> int:
        return self.col.count()

    def add(self, chunks: list[Chunk], batch: int = 64) -> None:
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            self.col.upsert(ids=[c.id for c in part], documents=[c.text for c in part],
                            embeddings=self.embedder.embed_documents([c.text for c in part]),
                            metadatas=[c.metadata for c in part])

    def query(self, text: str, k: int = 6, where: dict | None = None) -> list[Passage]:
        if self.count() == 0:
            return []
        res = self.col.query(query_embeddings=[self.embedder.embed_query(text)],
                             n_results=min(k, self.count()), where=where)
        out: list[Passage] = []
        for pid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append(Passage(id=pid, text=doc, file=meta["file"], part=meta["part"], macro=meta["macro"],
                               kind=meta["kind"], locator=meta.get("locator", ""),
                               stance=meta.get("stance", ""), score=round(1 - dist, 4)))
        return out
