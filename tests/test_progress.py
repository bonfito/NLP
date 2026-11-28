from conftest import FakeLLM
from test_graph import GOOD_QUIZ
from nlp_agent.graph import ask, build_graph
from nlp_agent.progress import ProgressStore
from nlp_agent.quizrun import run_quiz


def test_stats_and_weakest(tmp_path):
    p = ProgressStore(tmp_path / "p.json")
    assert "Non ho ancora" in p.summary_text()
    for ok in (True, False, False):
        p.record("Embeddings", ok)
    for _ in range(3):
        p.record("Tokenizzazione", True)
    assert p.stats()["Embeddings"]["accuracy"] == 1 / 3
    assert p.weakest() == ["Embeddings"]
    assert p.totals() == (4, 6)
    text = p.summary_text()
    assert "Da ripassare: Embeddings" in text and "6 domande" in text


def test_corrupted_file_is_ignored(tmp_path):
    f = tmp_path / "p.json"
    f.write_text("{non json", encoding="utf-8")
    p = ProgressStore(f)
    assert p.stats() == {}
    p.record("X", True)
    assert p.totals() == (1, 1)


def test_progress_route_needs_no_llm_answer(store, settings):
    tracker = ProgressStore(settings.progress_path)
    tracker.record("Embeddings", False)
    llm = FakeLLM(route="progress")
    out = ask(build_graph(llm, store, settings), "come sto andando?")
    assert "Embeddings" in out["answer"] and out["sources"] == []
    assert llm.calls == ["Sei il router di un assistente"[:30]] or len(llm.calls) == 1   # solo il router


def test_quiz_returns_structured_items(store, settings):
    out = ask(build_graph(FakeLLM(route="quiz", quiz=GOOD_QUIZ), store, settings), "quiz sulla self-attention")
    it = out["quiz_items"][0]
    assert it["correct"] == 1 and it["macro"] and it["source"]["file"]


def test_run_quiz_records_outcome(store, settings, tmp_path):
    out = ask(build_graph(FakeLLM(route="quiz", quiz=GOOD_QUIZ), store, settings), "quiz")
    tracker = ProgressStore(tmp_path / "q.json")
    shown = []
    assert run_quiz(out["quiz_items"], tracker, ask_input=lambda _: "b", show=shown.append) == (1, 1)
    assert tracker.totals() == (1, 1)
    run_quiz(out["quiz_items"], tracker, ask_input=lambda _: "a", show=shown.append)
    assert tracker.totals() == (1, 2)
