"""Quiz interattivo da terminale: fa le domande una alla volta, corregge e registra l'esito nei progressi."""
from __future__ import annotations

from typing import Callable

from .progress import ProgressStore


def run_quiz(items: list[dict], tracker: ProgressStore, ask_input: Callable[[str], str] = input,
             show: Callable[[str], None] = print) -> tuple[int, int]:
    """Ritorna (corrette, totali). Una risposta non valida conta come sbagliata; Ctrl+C interrompe senza perdere i dati."""
    ok = 0
    for n, it in enumerate(items, 1):
        show(f"\nDomanda {n}/{len(items)}: {it['question']}")
        for i, o in enumerate(it["options"]):
            show(f"   {'ABCD'[i]}) {o}")
        try:
            raw = ask_input("Risposta (A-D): ").strip().upper()[:1]
        except (KeyboardInterrupt, EOFError):
            return ok, n - 1
        correct = raw == "ABCD"[it["correct"]]
        ok += correct
        tracker.record(it["macro"], correct, it["question"], it.get("file", ""))
        show(("Corretto. " if correct else f"Sbagliato: la risposta era {'ABCD'[it['correct']]}. ") + it["explanation"])
        s = it["source"]
        show(f"   Fonte: {s['file']} — {s['macro']}, {s['locator']}")
    show(f"\nRisultato: {ok}/{len(items)}.")
    return ok, len(items)
