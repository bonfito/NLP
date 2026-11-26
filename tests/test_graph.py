import json

from conftest import FakeLLM
from nlp_agent.graph import ask, build_graph

GOOD_QUIZ = json.dumps([{"question": "Cosa calcola la self-attention?", "options": ["a", "b", "c", "d"],
                         "correct": 1, "explanation": "Query-Key-Value.", "source": 1}])


def run(llm, store, settings, q="Cos'è la self-attention?"):
    return ask(build_graph(llm, store, settings), q)


def test_theory_answer_has_sources_from_code(store, settings):
    out = run(FakeLLM(answer="La self-attention usa Q, K, V [1]."), store, settings)
    assert out["route"] == "theory"
    assert "Fonti:" in out["answer"] and ".pdf" in out["answer"]
    assert out["sources"] and out["sources"][0]["file"]
    assert out["revisions"] == 0


def test_offtopic_skips_retrieval(store, settings):
    llm = FakeLLM(route="offtopic")
    out = run(llm, store, settings, "Che tempo fa a Bologna?")
    assert "corso di NLP" in out["answer"]
    assert out["sources"] == []


def test_judge_triggers_rewrite_then_passes(store, settings):
    bad = {"grounded": False, "issues": "formula inventata", "new_query": "self-attention query key value"}
    ok = {"grounded": True, "issues": "", "new_query": ""}
    out = run(FakeLLM(judge=[bad, ok]), store, settings)
    assert out["revisions"] == 1 and out["corrections"] == 1
    assert "⚠️" not in out["answer"]
    assert out["query"] == "self-attention query key value"


def test_judge_loop_is_bounded_and_warns(store, settings):
    bad = {"grounded": False, "issues": "x", "new_query": ""}
    out = run(FakeLLM(judge=[bad]), store, settings)
    assert out["revisions"] == settings.max_revisions
    assert "⚠️" in out["answer"]


def test_unreadable_judge_does_not_block(store, settings):
    llm = FakeLLM()
    llm.judge_seq = ["non json"]          # json.dumps("non json") non è un dict -> ignorato
    out = run(llm, store, settings)
    assert out["revisions"] == 0 and "⚠️" not in out["answer"]


def test_quiz_valid_and_invalid(store, settings):
    out = run(FakeLLM(route="quiz", quiz=GOOD_QUIZ), store, settings, "fammi un quiz sulla self-attention")
    assert "Risposta: B" in out["answer"] and "Fonte:" in out["answer"]
    bad = run(FakeLLM(route="quiz", quiz="boh"), store, settings, "quiz")
    assert "Non sono riuscito" in bad["answer"]


def test_professor_opinion_marked_in_prompt(store, settings):
    seen = {}

    class Spy(FakeLLM):
        def complete(self, system, user, json_out=False):
            if "assistente di studio" in system and "PASSAGGI" in user:
                seen["user"] = user
            return super().complete(system, user)

    run(Spy(), store, settings, "Cos'è RLHF con reward model e PPO?")
    assert "[TESI DEL DOCENTE]" in seen["user"]


def test_empty_index_gives_clear_message(settings):
    from nlp_agent.embeddings import HashEmbedder
    from nlp_agent.store import VectorStore
    s = VectorStore(settings.index_dir / "empty", HashEmbedder())
    out = run(FakeLLM(), s, settings)
    assert "Non ho trovato" in out["answer"]


def test_extract_json_keeps_single_item_list():
    from nlp_agent.llm import extract_json
    assert extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
    assert extract_json('testo {"route": "quiz"} fine') == {"route": "quiz"}
    assert extract_json("niente json") is None


def test_router_query_is_english_search_query(store, settings):
    out = run(FakeLLM(), store, settings, "Cos'è la self-attention?")
    assert out["query"] == "self-attention query key value"      # non la domanda italiana grezza
    assert out["question"] == "Cos'è la self-attention?"


def test_router_falls_back_to_question_without_query_en(store, settings):
    class NoQuery(FakeLLM):
        def complete(self, system, user, json_out=False):
            if "router" in system:
                return '{"route": "theory"}'
            return super().complete(system, user)
    out = run(NoQuery(), store, settings, "Cos'è la self-attention?")
    assert out["query"] == "Cos'è la self-attention?"


def test_model_written_sources_are_stripped_and_replaced(store, settings):
    llm = FakeLLM(answer="La self-attention usa Q, K, V [1].\n\nFonti:\n- [5] Inventato.pdf (slide 99)")
    out = run(llm, store, settings)
    assert "Inventato.pdf" not in out["answer"]
    assert out["answer"].count("Fonti:") == 1


def test_diversify_caps_chunks_per_file():
    from nlp_agent.graph import diversify
    ps = [{"file": "Long.pdf", "id": i} for i in range(5)] + [{"file": "Slide.pdf", "id": 9}]
    out = diversify(ps, k=4, per_file=2)
    assert [p["file"] for p in out] == ["Long.pdf", "Long.pdf", "Slide.pdf"]


def test_diversify_guarantees_slides():
    from nlp_agent.graph import diversify
    arts = [{"file": f"Art{i}.pdf", "kind": "article", "id": i} for i in range(6)]
    slides = [{"file": "DNNs.pdf", "kind": "slide", "id": 20}, {"file": "Emb.pdf", "kind": "slide", "id": 21},
              {"file": "X.pdf", "kind": "slide", "id": 22}]
    out = diversify(arts + slides, k=4, min_slides=2)
    assert len(out) == 4
    assert sum(p["kind"] == "slide" for p in out) == 2
    assert [p["id"] for p in out[:2]] == [0, 1]           # i migliori restano
    assert diversify(arts, k=4, min_slides=2) == arts[:4]  # senza slide candidate non cambia nulla


def test_markdown_heading_sources_without_colon_are_stripped():
    from nlp_agent.graph import strip_model_sources
    t = "Spiegazione [1].\n\n### Fonti\n- [1] A.pdf\n- [DNNs.pdf (slide)]"
    assert strip_model_sources(t) == "Spiegazione [1]."
    assert strip_model_sources("Testo.\n\n**Fonti:** [1] A.pdf") == "Testo."
    assert strip_model_sources("Le Fonti dei dati sono varie.\nAltro.") == "Le Fonti dei dati sono varie.\nAltro."


def test_no_inline_markers_lists_all_retrieved_passages(store, settings):
    out = run(FakeLLM(answer="Risposta senza alcun marcatore."), store, settings,
              "self-attention query key value word2vec")
    assert "il modello non ha indicato quali ha usato" in out["answer"]
    assert len(out["sources"]) == len(out["passages"]) >= 2


def test_format_passages_only_keeps_original_numbers():
    from nlp_agent.nodes.helpers import format_passages
    ps = [{"file": f"F{i}.pdf", "kind": "slide", "macro": "m", "locator": "slide 1", "text": f"t{i}"} for i in range(1, 5)]
    out = format_passages(ps, only=[2, 4])
    assert "[2] F2.pdf" in out and "[4] F4.pdf" in out and "F1.pdf" not in out and "F3.pdf" not in out


def test_console_safe_replaces_unprintable_typography():
    from nlp_agent.cli import _console_safe
    assert _console_safe("self\u2011attention \u2212 uno\u202fdue") == "self-attention - uno due"
    assert _console_safe("perché è così") == "perché è così"


def test_strip_label_removes_duplicate_option_letters():
    from nlp_agent.graph import _strip_label
    assert _strip_label("A) Le prestazioni migliorano") == "Le prestazioni migliorano"
    assert _strip_label("b. testo") == "testo"
    assert _strip_label("Attention is all you need") == "Attention is all you need"


def test_slides_get_higher_per_file_cap():
    from nlp_agent.graph import diversify
    ps = [{"file": "Deck.pdf", "kind": "slide", "id": i} for i in range(4)] + [{"file": "Art.pdf", "kind": "article", "id": 9}]
    out = diversify(ps, k=6)
    assert sum(p["file"] == "Deck.pdf" for p in out) == 4


def test_professor_opinion_note_added_by_code():
    from nlp_agent.graph import opinion_note
    assert "docente" in opinion_note("Risposta senza menzione [1].", [{"stance": "professor_opinion"}])
    assert opinion_note("Secondo il docente, X [1].", [{"stance": "professor_opinion"}]) == ""
    assert opinion_note("Risposta [1].", [{"stance": ""}]) == ""
    assert "[2]" in opinion_note("A [1] e B [2].", [{"stance": ""}, {"stance": "professor_opinion"}])


def test_bm25_finds_exact_rare_terms():
    from nlp_agent.store import BM25
    docs = ["energy of language models and their training cost", "LHC at CERN uses 1.3 TWh per year",
            "language models predict the next word"]
    bm = BM25(["a", "b", "c"], docs)
    assert bm.top("consumption compared to CERN", 2)[0] == "b"
    assert bm.top("CERN", 1) == ["b"]
    assert bm.top("zzzz", 3) == []


def test_hybrid_query_returns_lexical_match_and_respects_k(store):
    out = store.query("CERN electricity", k=3)
    assert len(out) <= 3
    assert store.query("self-attention", k=2, hybrid=False)
