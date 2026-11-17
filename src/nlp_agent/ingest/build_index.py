"""Costruisce l'indice: legge data/raw, classifica con course_map.yaml, chunk, embedding, Chroma."""
from __future__ import annotations

import sys

from ..config import Settings
from ..course_map import CourseMap
from ..embeddings import get_embedder
from ..store import VectorStore
from .chunking import chunk_document
from .loaders import load_file

SUPPORTED = {".pdf", ".html", ".htm"}


def build_index(settings: Settings, ocr: bool = True, only: list[str] | None = None, reset: bool = True) -> dict:
    cmap = CourseMap.load(settings.course_map)
    store = VectorStore(settings.index_dir, get_embedder(settings.embedding_model))
    if reset and not only:
        store.reset()

    report = {"indexed": {}, "skipped": [], "unmapped": [], "warnings": {}}
    for path in sorted(settings.raw_dir.iterdir()):
        if path.suffix.lower() not in SUPPORTED or (only and path.name not in only):
            continue
        if cmap.is_excluded(path.name):
            report["skipped"].append(f"{path.name} (escluso da course_map.yaml)")
            continue
        info = cmap.lookup(path.name)
        if info is None:
            report["unmapped"].append(path.name)
            continue
        print(f"[ingest] {path.name} ...", file=sys.stderr, flush=True)
        doc = load_file(path, info.kind, ocr=ocr, cache_dir=settings.index_dir.parent / "ocr_cache")
        if doc.warnings:
            report["warnings"][path.name] = doc.warnings
        chunks = chunk_document(path.name, doc, info)
        if chunks:
            store.add(chunks)
        report["indexed"][path.name] = len(chunks)
    report["total_chunks"] = store.count()
    return report
