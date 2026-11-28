"""Interfaccia web: `streamlit run app/streamlit_app.py` (dalla radice del repo)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

from nlp_agent.config import Settings
from nlp_agent.graph import ask, build_graph
from nlp_agent.progress import ProgressStore

st.set_page_config(page_title="Assistente NLP", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Carico indice e modelli...")
def load_agent():
    from nlp_agent.embeddings import get_embedder
    from nlp_agent.llm import get_llm
    from nlp_agent.store import VectorStore

    s = Settings.from_env()
    store = VectorStore(s.index_dir, get_embedder(s.embedding_model))
    if store.count() == 0:
        return None, s
    return build_graph(get_llm(s), store, s), s


app, settings = load_agent()


@st.cache_data
def macro_topics(path: str) -> list[str]:
    import yaml

    files = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("files", {})
    return sorted({v["macro"] for v in files.values() if isinstance(v, dict) and v.get("macro")})


def run_question(q: str) -> None:
    st.session_state.chat.append(("user", q, []))
    with st.spinner("Cerco nei materiali..."):
        state = ask(app, q, st.session_state.history)
    st.session_state.history = state["messages"]
    if state.get("quiz_items"):
        st.session_state.quiz = {"items": state["quiz_items"], "i": 0, "done": {}}
        st.session_state.chat.append(("assistant", "Ecco il quiz: rispondi cliccando le opzioni qui sotto.", []))
    else:
        body = state["answer"].split("\n\nFonti:")[0]
        st.session_state.chat.append(("assistant", body, state.get("sources", [])))

tracker = ProgressStore(settings.progress_path)
if app is None:
    st.error("Indice vuoto: esegui prima `python -m nlp_agent.cli index`.")
    st.stop()

st.session_state.setdefault("history", [])    # messaggi per il grafo
st.session_state.setdefault("chat", [])       # (ruolo, testo, fonti)
st.session_state.setdefault("quiz", None)     # {"items": [...], "i": 0, "done": {i: scelta}}

with st.sidebar:
    st.header("Quiz")
    topic = st.selectbox("Argomento", macro_topics(str(settings.course_map)), index=None,
                         placeholder="Scegli un macroargomento")
    custom = st.text_input("...oppure un argomento libero", placeholder="es. attention, BPE, WordNet")
    weak = tracker.weakest()
    if weak and st.button(f"Quiz sul tuo punto debole: {weak[0]}"):
        st.session_state.pending = f"quiz su {weak[0]}"
    if st.button("Avvia quiz", type="primary", disabled=not (custom.strip() or topic)):
        st.session_state.pending = f"quiz su {custom.strip() or topic}"
    st.divider()
    st.header("I tuoi progressi")
    stats = tracker.stats()
    if not stats:
        st.caption("Nessun quiz ancora svolto.")
    else:
        ok, tot = tracker.totals()
        st.metric("Risposte corrette", f"{ok}/{tot}", f"{round(100 * ok / tot)}%")
        for m, v in sorted(stats.items(), key=lambda x: x[1]["accuracy"]):
            st.progress(v["accuracy"], text=f"{m}: {v['correct']}/{v['attempts']}")
        if weak:
            st.warning("Da ripassare: " + ", ".join(tracker.weakest()))
    if st.button("Azzera progressi"):
        tracker.reset()
        st.rerun()
    if st.button("Nuova conversazione"):
        st.session_state.update(history=[], chat=[], quiz=None)
        st.rerun()

st.title("Assistente di studio — NLP")
st.caption("Rispondo solo con i materiali del corso. Prova: «cos'è la self-attention?», «quiz su tokenization», "
           "«ripasso di embeddings», «cosa devo ripassare?».")

for role, text, sources in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)
        if sources:
            with st.expander("Fonti"):
                for n, e in enumerate(sources, 1):
                    st.markdown(f"**[{n}]** {e['file']} — {e['macro']}, {e['locator']}")

quiz = st.session_state.quiz
if quiz:
    items, i = quiz["items"], quiz["i"]
    if i < len(items):
        it = items[i]
        with st.chat_message("assistant"):
            st.markdown(f"**Domanda {i + 1}/{len(items)}** — {it['question']}")
            if i in quiz["done"]:
                choice = quiz["done"][i]
                for k, o in enumerate(it["options"]):
                    mark = "✅" if k == it["correct"] else ("❌" if k == choice else "▫️")
                    st.markdown(f"{mark} {'ABCD'[k]}) {o}")
                st.info(it["explanation"] + f"\n\nFonte: {it['source']['file']} — {it['source']['macro']}, "
                        f"{it['source']['locator']}")
                if st.button("Avanti" if i + 1 < len(items) else "Termina", key=f"next{i}"):
                    quiz["i"] += 1
                    st.rerun()
            else:
                for k, o in enumerate(it["options"]):
                    if st.button(f"{'ABCD'[k]}) {o}", key=f"opt{i}{k}"):
                        quiz["done"][i] = k
                        tracker.record(it["macro"], k == it["correct"], it["question"], it["file"])
                        st.rerun()
    else:
        ok = sum(quiz["done"][j] == items[j]["correct"] for j in range(len(items)))
        st.success(f"Quiz finito: {ok}/{len(items)} corrette. I risultati sono salvati nei progressi.")
        if st.button("Chiudi quiz"):
            st.session_state.quiz = None
            st.rerun()

q = st.chat_input("Scrivi una domanda...") or st.session_state.pop("pending", None)
if q:
    run_question(q)
    st.rerun()
