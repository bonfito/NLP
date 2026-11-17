"""Suddivisione in chunk con metadati per la citazione (file, macroargomento, locator)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..course_map import CourseFile
from .loaders import LoadedDoc


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + len(s) + 1 > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    final = []
    for piece in out:           # frasi enormi (es. codice): taglio duro
        while len(piece) > max_chars:
            final.append(piece[:max_chars])
            piece = piece[max_chars:]
        final.append(piece)
    return final


def _pack(blocks, target: int, max_chars: int, overlap: int):
    """Raggruppa blocchi consecutivi (testo, pagina, intestazione) in chunk di ~target caratteri."""
    chunks = []
    cur, pages, heads = "", [], []
    for text, page, head in blocks:
        for piece in _split_long(text, max_chars):
            if cur and len(cur) + len(piece) + 2 > max_chars:
                chunks.append((cur, pages, heads))
                tail = cur[-overlap:] if overlap else ""
                cur, pages, heads = tail, [], []
            cur = f"{cur}\n\n{piece}".strip()
            if page is not None:
                pages.append(page)
            if head:
                heads.append(head)
            if len(cur) >= target:
                chunks.append((cur, pages, heads))
                tail = cur[-overlap:] if overlap else ""
                cur, pages, heads = tail, [], []
    if cur.strip() and (not chunks or pages or heads):
        chunks.append((cur, pages, heads))
    return chunks


def _locator(kind: str, pages: list[int], heads: list[str]) -> tuple[str, int, int]:
    if pages:
        a, b = min(pages), max(pages)
        if kind == "slide":
            return (f"slide {a}" if a == b else f"slide {a}-{b}"), a, b
        return (f"p. {a}" if a == b else f"pp. {a}-{b}"), a, b
    if heads:
        return f"sezione «{heads[0]}»", -1, -1
    return "", -1, -1


def chunk_document(filename: str, doc: LoadedDoc, info: CourseFile) -> list[Chunk]:
    """Slide: slide consecutive unite fino a ~700 caratteri (molte sono quasi vuote).
    Articoli/libri/tutorial: paragrafi in ~1100 caratteri con piccola sovrapposizione."""
    is_slide = info.kind == "slide"
    target, max_chars, overlap = (700, 1600, 0) if is_slide else (1100, 1500, 150)

    blocks = []
    for u in doc.units:
        if is_slide or not u.blocks:
            blocks.append((u.text, u.page, u.heading))
        else:
            blocks.extend((b, u.page, u.heading) for b in u.blocks)

    out = []
    for n, (text, pages, heads) in enumerate(_pack(blocks, target, max_chars, overlap)):
        text = text.strip()
        if len(text) < 30:
            continue
        loc, p0, p1 = _locator(info.kind, pages, heads)
        out.append(Chunk(id=f"{filename}::{n:04d}", text=text,
                         metadata=dict(file=filename, part=info.part, macro=info.macro, kind=info.kind,
                                       stance=info.stance, locator=loc, page_start=p0, page_end=p1)))
    return out
