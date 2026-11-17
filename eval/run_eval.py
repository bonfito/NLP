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
from nlp_agent.evaluation import run_eval, to_markdown
from nlp_agent.graph import build_graph
from nlp_agent.llm import get_llm
from nlp_agent.store import VectorStore

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int)
ap.add_argument("--no-baseline", action="store_true")
ap.add_argument("--questions", default="eval/questions.yaml")
args = ap.parse_args()

s = Settings.from_env()
questions = yaml.safe_load(Path(args.questions).read_text(encoding="utf-8"))[: args.limit]
store = VectorStore(s.index_dir, get_embedder(s.embedding_model))
if store.count() == 0:
    sys.exit("Indice vuoto: esegui prima `python -m nlp_agent.cli index`.")
llm = get_llm(s)
result = run_eval(build_graph(llm, store, s), llm, questions, baseline=not args.no_baseline)

out = Path("eval/results"); out.mkdir(parents=True, exist_ok=True)
path = out / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(to_markdown(result["summary"]))
print(f"\nDettaglio per domanda: {path}")
