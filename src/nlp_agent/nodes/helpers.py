"""Formattazione dei passaggi e delle fonti (la citazione è costruita dal codice, non dal modello)."""
from __future__ import annotations

import re

KIND_IT = {"slide": "slide", "article": "articolo", "tutorial": "tutorial", "book": "libro/capitolo"}


def format_passages(passages: list[dict]) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        tag = " [TESI DEL DOCENTE]" if p.get("stance") == "professor_opinion" else ""
        head = f"[{i}] {p['file']} ({KIND_IT.get(p['kind'], p['kind'])}, {p['macro']}, {p['locator']}){tag}"
        blocks.append(f"{head}\n{p['text']}")
    return "\n\n".join(blocks)


def cited_indices(text: str, n: int) -> list[int]:
    seen = []
    for m in re.finditer(r"\[(\d{1,2})\]", text):
        i = int(m.group(1))
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
    idx = cited_indices(answer, len(passages))
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
