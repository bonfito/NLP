import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nlp_agent.config import Settings
from nlp_agent.embeddings import HashEmbedder
from nlp_agent.ingest.chunking import Chunk
from nlp_agent.store import VectorStore


class FakeLLM:
    """LLM finto: risponde in base al system prompt. `script` permette di forzare le risposte per nodo."""

    def __init__(self, route="theory", judge=None, quiz=None, answer="Risposta di prova [1]."):
        self.route, self.answer = route, answer
        self.judge_seq = list(judge or [{"grounded": True, "issues": "", "new_query": ""}])
        self.quiz = quiz
        self.calls = []

    def complete(self, system: str, user: str, json_out: bool = False) -> str:
        self.calls.append(system[:30])
        if "router" in system:
            return json.dumps({"route": self.route, "topic": "self-attention", "query_en": "self-attention query key value"})
        if "verificatore" in system:
            v = self.judge_seq.pop(0) if len(self.judge_seq) > 1 else self.judge_seq[0]
            return json.dumps(v)
        if "generatore di quiz" in system:
            return self.quiz if self.quiz is not None else "non json"
        if "Riformula" in system:
            return "self attention transformer"
        return self.answer


@pytest.fixture
def settings(tmp_path):
    return Settings(index_dir=tmp_path / "idx", top_k=3, max_revisions=2, progress_path=tmp_path / "progress.json")


@pytest.fixture
def store(settings):
    s = VectorStore(settings.index_dir, HashEmbedder())
    meta = dict(part="Part I", kind="slide", stance="", page_start=1, page_end=1)
    s.add([
        Chunk("a::0", "self-attention computes query key value projections of the input embeddings",
              dict(meta, file="DNNs.pdf", macro="Deep learning", locator="slide 19")),
        Chunk("b::0", "word2vec CBOW and skip-gram predict words from context vectors",
              dict(meta, file="Embeddings.pdf", macro="Embeddings", locator="slide 11")),
        Chunk("c::0", "RLHF uses a reward model and PPO to please the judge",
              dict(meta, file="SOTA4LLMs2026.pdf", macro="Stato dell'arte LLM", locator="slide 4",
                   stance="professor_opinion", part="Part III")),
    ])
    return s
