import json

from conftest import FakeLLM
from nlp_agent.evaluation import run_eval, summarize
from nlp_agent.graph import build_graph


class EvalLLM(FakeLLM):
    """Correttore finto: 'corretta' se la risposta valutata contiene 'attention'. La baseline risponde 'Non so.'"""

    def complete(self, system, user):
        if "correttore" in system:
            return json.dumps({"correct": "attention" in user.split("RISPOSTA DA VALUTARE")[1].lower()})
        if system.startswith("Sei un assistente per il corso di Natural"):
            return "Non so."
        return super().complete(system, user)


QS = [
    {"id": "t1", "question": "Cos'è la self-attention?", "expected_answer": "attention",
     "expected_macro": "Deep learning", "expected_files": ["DNNs.pdf"]},
    {"id": "t2", "question": "Cos'è la self-attention?", "expected_answer": "attention",
     "expected_macro": "Embeddings", "expected_files": ["Embeddings.pdf"]},   # fonte attesa sbagliata apposta
    {"id": "o1", "question": "Ricetta carbonara", "route": "offtopic", "expected_answer": "rifiuto"},
]


def test_agent_vs_baseline_and_source_accuracy(store, settings):
    llm = EvalLLM(answer="La self-attention usa Q, K, V [1].")
    res = run_eval(build_graph(llm, store, settings), llm, QS)
    t1, t2, o1 = res["rows"]
    assert t1["correct"] is True and t1["baseline_correct"] is False
    assert t1["source_file_ok"] is True and t1["source_macro_ok"] is True
    assert t2["source_file_ok"] is False and t2["source_macro_ok"] is False   # fonte indicata ≠ fonte attesa
    s = res["summary"]
    assert s["agent_correct"] == 1.0 and s["baseline_correct"] == 0.0
    assert s["source_file_accuracy"] == 0.5


def test_offtopic_refusal_measured(store, settings):
    llm = EvalLLM(route="offtopic")
    res = run_eval(build_graph(llm, store, settings), llm, [QS[2]])
    assert res["rows"][0]["route_ok"] is True and res["summary"]["offtopic_refusal"] == 1.0


def test_summarize_handles_empty():
    s = summarize([])
    assert s["agent_correct"] is None and s["total_judge_corrections"] == 0
