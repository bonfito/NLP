"""CLI: `python -m nlp_agent.cli index | chat | ask "domanda"`."""
from __future__ import annotations

import argparse
import json
import sys

from .config import Settings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nlp_agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_idx = sub.add_parser("index", help="costruisce l'indice da data/raw")
    p_idx.add_argument("--no-ocr", action="store_true")
    p_idx.add_argument("--only", nargs="*")
    sub.add_parser("chat", help="chat interattiva")
    p_ask = sub.add_parser("ask", help="una domanda singola")
    p_ask.add_argument("question")
    args = ap.parse_args(argv)
    s = Settings.from_env()

    if args.cmd == "index":
        from .ingest.build_index import build_index

        rep = build_index(s, ocr=not args.no_ocr, only=args.only, reset=not args.only)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
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

    if args.cmd == "ask":
        print(ask(app, args.question)["answer"])
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
            print("\n" + state["answer"])


if __name__ == "__main__":
    sys.exit(main())
