"""Formattazione dei passaggi e delle fonti (la citazione è costruita dal codice, non dal modello)."""
from __future__ import annotations

import re

KIND_IT = {"slide": "slide", "article": "articolo", "tutorial": "tutorial", "book": "libro/capitolo"}


def format_passages(passages: list[dict], only: list[int] | None = None) -> str:
    """Passaggi numerati da 1. Con `only` (numeri 1-based) include solo quelli, mantenendo la numerazione originale."""
    blocks = []
    for i, p in enumerate(passages, 1):
        if only is not None and i not in only:
            continue
        tag = " [TESI DEL DOCENTE]" if p.get("stance") == "professor_opinion" else ""
        head = f"[{i}] {p['file']} ({KIND_IT.get(p['kind'], p['kind'])}, {p['macro']}, {p['locator']}){tag}"
        blocks.append(f"{head}\n{p['text']}")
    return "\n\n".join(blocks)


def normalize_citations(text: str) -> str:
    """Alcuni modelli scrivono le citazioni come 【1】 o （1）: le riporto al formato [1] che il codice riconosce."""
    return text.replace("\u3010", "[").replace("\u3011", "]")


def cited_indices(text: str, n: int) -> list[int]:
    """Numeri dei passaggi citati: riconosce [1], [1][2] e [1, 2]."""
    seen = []
    for m in re.finditer(r"\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]", normalize_citations(text)):
        for num in re.findall(r"\d{1,2}", m.group(1)):
            i = int(num)
            if 1 <= i <= n and i not in seen:
                seen.append(i)
    return seen


def source_entry(p: dict) -> dict:
    return {"file": p["file"], "kind": KIND_IT.get(p["kind"], p["kind"]), "part": p["part"],
            "macro": p["macro"], "locator": p["locator"]}


def sources_block(answer: str, passages: list[dict]) -> tuple[str, list[dict]]:
    """Elenco 'Fonti' costruito dal codice.
    Se la risposta contiene marcatori [n], elenca quei passaggi. Altrimenti (modelli piccoli che non citano)
    elenca TUTTI i passaggi consultati, con una nota: meglio una fonte in più che una fonte nascosta."""
    idx = sorted(cited_indices(answer, len(passages)))
    note = ""
    if not idx:
        idx = list(range(1, len(passages) + 1))
        note = "(passaggi consultati: il modello non ha indicato quali ha usato)\n"
    entries, lines = [], []
    for i in idx:
        e = source_entry(passages[i - 1])
        entries.append(e)
        lines.append(f"[{i}] {e['file']} — {e['macro']} ({e['part']}), {e['kind']}, {e['locator']}")
    return note + "\n".join(lines), entries
