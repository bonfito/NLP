"""Mostra cosa trova il retrieval per una query (debug): `python scripts/show_retrieval.py "query in inglese" [k]`."""
import sys

from nlp_agent.config import Settings
from nlp_agent.embeddings import get_embedder
from nlp_agent.graph import CANDIDATES_FACTOR, diversify
from nlp_agent.store import VectorStore

query = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else Settings.from_env().top_k
s = Settings.from_env()
store = VectorStore(s.index_dir, get_embedder(s.embedding_model))
cands = store.query(query, k=k * CANDIDATES_FACTOR)
chosen = {p["id"] for p in diversify(cands, k)}
for i, p in enumerate(cands, 1):
    mark = "*" if p["id"] in chosen else " "
    print(f"{mark}{i:2d} {p['score']:.3f} {p['file']} [{p['kind']}] {p['locator']} :: {p['text'][:90]!r}")
print("\n* = passaggi passati al modello")
