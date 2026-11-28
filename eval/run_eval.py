"""Esegue la valutazione: `python eval/run_eval.py [--limit N] [--no-baseline]`. Richiede indice e LLM configurati."""
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from nlp_agent.config import Settings
from nlp_agent.embeddings import get_embedder
from nlp_agent.evaluation import run_eval, select_questions, to_markdown
from nlp_agent.graph import build_graph
from nlp_agent.llm import get_llm
from nlp_agent.store import VectorStore

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int)
ap.add_argument("--no-baseline", action="store_true")
ap.add_argument("--ids", nargs="*", help="solo queste domande, es. --ids en-01 sota-01")
ap.add_argument("--checkpoint", default="eval/results/checkpoint.json",
                help="salva i risultati dopo ogni domanda e riparte da dove si era fermato (cancellalo per ricominciare)")
ap.add_argument("--questions", default="eval/questions.yaml")
args = ap.parse_args()

s = Settings.from_env()
questions = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))
if args.ids:
    questions = [q for q in questions if q["id"] in args.ids]
questions = select_questions(questions, args.limit)
store = VectorStore(s.index_dir, get_embedder(s.embedding_model))
if store.count() == 0:
    sys.exit("Indice vuoto: esegui prima `python -m nlp_agent.cli index`.")
llm = get_llm(s)
out = Path("eval/results"); out.mkdir(parents=True, exist_ok=True)
ckpt = Path(args.checkpoint)
wanted = {q["id"] for q in questions}
done = []
if ckpt.exists():
    done = [r for r in json.loads(ckpt.read_text(encoding="utf-8")).get("rows", []) if r["id"] in wanted]
    print(f"Checkpoint trovato: {len(done)} domande già fatte, riparto dalle altre.")


def save(rows):
    ckpt.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


result = run_eval(build_graph(llm, store, s), llm, questions, baseline=not args.no_baseline,
                  done_rows=done, on_row=save)
path = out / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(to_markdown(result["summary"]))
print(f"\nDettaglio per domanda: {path}")
if result["interrupted"]:
    print(f"\nINTERROTTA ({len(result['rows'])}/{len(questions)} fatte) a causa di: {result['interrupted']}\n"
          "Probabile limite giornaliero Groq: rilancia lo stesso comando più tardi, riparte dal checkpoint.")
    sys.exit(2)
