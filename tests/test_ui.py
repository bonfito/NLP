import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from conftest import FakeLLM
from test_graph import GOOD_QUIZ


def test_ui_chat_and_quiz_updates_progress(store, settings, monkeypatch):
    import nlp_agent.embeddings as emb
    import nlp_agent.llm as llm_mod
    from nlp_agent.progress import ProgressStore

    monkeypatch.setenv("INDEX_DIR", str(settings.index_dir))
    monkeypatch.setenv("PROGRESS_PATH", str(settings.progress_path))
    monkeypatch.setattr(emb, "get_embedder", lambda name: emb.HashEmbedder())
    fake = FakeLLM(route="quiz", quiz=GOOD_QUIZ)
    monkeypatch.setattr(llm_mod, "get_llm", lambda s: fake)

    at = AppTest.from_file("../app/streamlit_app.py", default_timeout=30).run()
    assert not at.exception
    at.chat_input[0].set_value("quiz sulla self-attention").run()
    assert not at.exception
    opts = [b for b in at.button if b.label.startswith("B)")]
    assert opts
    opts[0].click().run()
    assert ProgressStore(settings.progress_path).totals() == (1, 1)
