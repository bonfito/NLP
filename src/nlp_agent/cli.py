"""CLI: `python -m nlp_agent.cli index | chat | ask "domanda"`."""
from __future__ import annotations

import argparse
import json
import sys

from .config import Settings


_SAFE = {"\u2011": "-", "\u2212": "-", "\u202f": " ", "\u2009": " ", "\u200b": ""}


def _console_safe(text: str) -> str:
    """La console di Windows (cp1252/cp850) non stampa alcuni caratteri tipografici che i modelli usano
    (trattino non separabile, segno meno, spazi sottili): li sostituisco, il resto passa invariato."""
    return text.translate(str.maketrans(_SAFE))


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")    # un carattere non stampabile non deve far crollare il programma
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="nlp_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_idx = sub.add_parser("index", help="costruisce l'indice da data/raw")
    p_idx.add_argument("--no-ocr", action="store_true")
    p_idx.add_argument("--only", nargs="*")
    sub.add_parser("chat", help="chat interattiva")
    p_quiz = sub.add_parser("quiz", help="quiz interattivo con correzione e tracciamento dei progressi")
    p_quiz.add_argument("topic")
    sub.add_parser("progress", help="mostra i risultati dei quiz e cosa ripassare")
    p_ask = sub.add_parser("ask", help="una domanda singola")
    p_ask.add_argument("question")
    args = ap.parse_args(argv)
    s = Settings.from_env()

    if args.cmd == "index":
        from .ingest.build_index import build_index

        rep = build_index(s, ocr=not args.no_ocr, only=args.only, reset=not args.only)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "progress":
        from .progress import ProgressStore

        print(_console_safe(ProgressStore(s.progress_path).summary_text()))
        return 0

    from .embeddings import get_embedder
    from .graph import ask, build_graph
    from .llm import get_llm
    from .store import VectorStore

    store = VectorStore(s.index_dir, get_embedder(s.embedding_model))
    if store.count() == 0:
        print("Indice vuoto: esegui prima `python -m nlp_agent.cli index`.", file=sys.stderr)
        return 1
    app = build_graph(get_llm(s), store, s)

    if args.cmd == "quiz":
        from .progress import ProgressStore
        from .quizrun import run_quiz

        state = ask(app, f"quiz su {args.topic}")
        items = state.get("quiz_items") or []
        if not items:
            print(_console_safe(state["answer"]))
            return 1
        run_quiz(items, ProgressStore(s.progress_path), show=lambda t: print(_console_safe(t)))
        return 0
    if args.cmd == "ask":
        print(_console_safe(ask(app, args.question)["answer"]))
        return 0
    history: list[dict] = []
    print("Assistente NLP — scrivi una domanda, 'quiz su <argomento>', 'ripasso di <argomento>'. Ctrl+C per uscire.")
    while True:
        try:
            q = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            return 0
        if q:
            state = ask(app, q, history)
            history = state["messages"]
            print("\n" + _console_safe(state["answer"]))


if __name__ == "__main__":
    sys.exit(main())
