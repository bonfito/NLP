"""Grafo LangGraph: router -> retrieve -> (answer | review | quiz) -> judge -> (fine | rewrite -> retrieve)."""
from __future__ import annotations

import re

from langgraph.graph import END, START, StateGraph

from . import prompts
from .config import Settings
from .llm import LLM, extract_json
from .nodes.helpers import format_passages, sources_block
from .state import AgentState
from .store import VectorStore

OFFTOPIC_MSG = ("Posso aiutarti solo su argomenti del corso di NLP (embeddings, Transformer, prompting, "
                "parsing, ecc.). Prova a riformulare la domanda in relazione al corso.")
NO_PASSAGES_MSG = "Non ho trovato nei materiali del corso passaggi pertinenti a questa richiesta."
VALID_ROUTES = {"theory", "quiz", "review", "offtopic"}
MAX_PER_FILE = 2          # al massimo 2 chunk per file tra i passaggi: evita che un solo documento lungo affolli i risultati
CANDIDATES_FACTOR = 4     # si recuperano k*4 candidati e poi si applica il limite per file

_MODEL_SOURCES = re.compile(
    r"\n[ \t]*(?:#{1,6}[ \t]*)?\**[ \t]*(?:Fonti|Fonte|Sources?|Riferimenti|Bibliografia)[ \t]*\**[ \t]*(?::.*|(?:\n.*)?)\Z",
    re.IGNORECASE | re.DOTALL,
)


def strip_model_sources(text: str) -> str:
    """Toglie l'elenco 'Fonti:' eventualmente scritto dal modello: le fonti le costruisce il codice dai metadati."""
    return _MODEL_SOURCES.sub("", text).rstrip()


MIN_SLIDES = 2   # le slide hanno poco testo e perdono contro articoli lunghi: ne garantisco almeno 2 se pertinenti


def diversify(passages: list[dict], k: int, per_file: int = MAX_PER_FILE, min_slides: int = MIN_SLIDES) -> list[dict]:
    """Prende i migliori k passaggi con al massimo `per_file` chunk dallo stesso file (ordine di similarità conservato).
    Se tra i candidati ci sono slide ma meno di `min_slides` nella selezione, le migliori slide scartate prendono il
    posto dei passaggi non-slide con punteggio più basso."""
    out, rest, count = [], [], {}
    for p in passages:
        if count.get(p["file"], 0) < per_file and len(out) < k:
            out.append(p)
            count[p["file"]] = count.get(p["file"], 0) + 1
        else:
            rest.append(p)
    slides_in = sum(p.get("kind") == "slide" for p in out)
    extra = [p for p in rest if p.get("kind") == "slide"]
    while slides_in < min_slides and extra:
        victim = next((i for i in range(len(out) - 1, -1, -1) if out[i].get("kind") != "slide"), None)
        if victim is None:
            break
        out.pop(victim)
        out.append(extra.pop(0))
        slides_in += 1
    return out


def _heuristic_route(q: str) -> str:
    ql = q.lower()
    if "quiz" in ql or "scelta multipla" in ql or "risposta multipla" in ql:
        return "quiz"
    if any(w in ql for w in ("ripasso", "ripassa", "riassum", "review")):
        return "review"
    return "theory"


def build_graph(llm: LLM, store: VectorStore, settings: Settings):
    """Costruisce e compila il grafo. llm e store sono iniettati, così si possono testare con dei finti."""

    def router(state: AgentState) -> dict:
        q = state["question"]
        data = extract_json(llm.complete(prompts.ROUTER_SYSTEM, q, json_out=True))
        route = data.get("route") if isinstance(data, dict) else None
        if route not in VALID_ROUTES:
            route, data = _heuristic_route(q), {}
        data = data if isinstance(data, dict) else {}
        topic = data.get("topic") or q
        query_en = (data.get("query_en") or "").strip()
        query = topic if route in ("quiz", "review") else (query_en or q)
        return {"route": route, "topic": topic, "query": query,
                "revisions": 0, "corrections": 0, "passages": [], "verdict": {}}

    def offtopic(state: AgentState) -> dict:
        return {"answer": OFFTOPIC_MSG, "messages": [{"role": "assistant", "content": OFFTOPIC_MSG}], "sources": []}

    def retrieve(state: AgentState) -> dict:
        k = settings.top_k * (2 if state["route"] in ("quiz", "review") else 1)
        return {"passages": diversify(store.query(state["query"], k=k * CANDIDATES_FACTOR), k)}

    def _finish(state: AgentState, text: str) -> dict:
        text = strip_model_sources(text)
        block, entries = sources_block(text, state["passages"])
        full = f"{text}\n\nFonti:\n{block}"
        return {"answer": full, "sources": entries, "messages": [{"role": "assistant", "content": full}]}

    def answer(state: AgentState) -> dict:
        if not state["passages"]:
            return {"answer": NO_PASSAGES_MSG, "sources": [],
                    "messages": [{"role": "assistant", "content": NO_PASSAGES_MSG}]}
        system = prompts.REVIEW_SYSTEM if state["route"] == "review" else prompts.ANSWER_SYSTEM
        user = f"PASSAGGI:\n{format_passages(state['passages'])}\n\nRICHIESTA: {state['question']}"
        return _finish(state, llm.complete(system, user).strip())

    def judge(state: AgentState) -> dict:
        if not state["passages"]:
            return {"verdict": {"grounded": True, "issues": "", "new_query": ""}}
        user = (f"DOMANDA: {state['question']}\n\nPASSAGGI:\n{format_passages(state['passages'])}\n\n"
                f"RISPOSTA:\n{state['answer'].split(chr(10) + chr(10) + 'Fonti:')[0]}")
        data = extract_json(llm.complete(prompts.JUDGE_SYSTEM, user, json_out=True))
        if not isinstance(data, dict) or "grounded" not in data:
            data = {"grounded": True, "issues": "", "new_query": ""}   # judge illeggibile: non blocco la risposta
        out = {"verdict": data}
        if not data["grounded"] and state["revisions"] >= settings.max_revisions:
            note = ("\n\n⚠️ Attenzione: la verifica automatica non ha confermato che tutta la risposta sia "
                    f"sostenuta dai materiali ({data.get('issues', '')}). Controlla le fonti.")
            out["answer"] = state["answer"] + note
        return out

    def after_judge(state: AgentState) -> str:
        v = state["verdict"]
        return "rewrite" if (not v.get("grounded", True) and state["revisions"] < settings.max_revisions) else "end"

    def rewrite(state: AgentState) -> dict:
        v = state["verdict"]
        new_q = (v.get("new_query") or "").strip() or llm.complete(prompts.REWRITE_SYSTEM, state["question"]).strip()
        return {"query": new_q, "revisions": state["revisions"] + 1, "corrections": state["corrections"] + 1}

    def quiz(state: AgentState) -> dict:
        if not state["passages"]:
            return {"answer": NO_PASSAGES_MSG, "sources": [],
                    "messages": [{"role": "assistant", "content": NO_PASSAGES_MSG}]}
        user = f"PASSAGGI:\n{format_passages(state['passages'])}\n\nRICHIESTA: {state['question']}"
        items = None
        for _ in range(2):                      # un secondo tentativo se il JSON non è valido
            data = extract_json(llm.complete(prompts.QUIZ_SYSTEM, user))
            items = [d for d in data if _valid_quiz_item(d)] if isinstance(data, list) else None
            if items:
                break
        if not items:
            msg = "Non sono riuscito a generare un quiz valido. Riprova indicando meglio l'argomento."
            return {"answer": msg, "sources": [], "messages": [{"role": "assistant", "content": msg}]}
        lines, entries = [], []
        for n, it in enumerate(items, 1):
            src = state["passages"][min(max(int(it.get("source", 1)), 1), len(state["passages"])) - 1]
            from .nodes.helpers import source_entry
            e = source_entry(src)
            entries.append(e)
            opts = "\n".join(f"   {'ABCD'[i]}) {o}" for i, o in enumerate(it["options"]))
            lines.append(f"{n}. {it['question']}\n{opts}\n   Risposta: {'ABCD'[it['correct']]} — {it['explanation']}\n"
                         f"   Fonte: {e['file']} — {e['macro']}, {e['kind']}, {e['locator']}")
        text = "\n\n".join(lines)
        return {"answer": text, "sources": entries, "messages": [{"role": "assistant", "content": text}]}

    def route_after_router(state: AgentState) -> str:
        return "offtopic" if state["route"] == "offtopic" else "retrieve"

    def route_after_retrieve(state: AgentState) -> str:
        return "quiz" if state["route"] == "quiz" else "answer"

    g = StateGraph(AgentState)
    for name, fn in [("router", router), ("offtopic", offtopic), ("retrieve", retrieve), ("answer", answer),
                     ("judge", judge), ("rewrite", rewrite), ("quiz", quiz)]:
        g.add_node(name, fn)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_after_router, {"offtopic": "offtopic", "retrieve": "retrieve"})
    g.add_conditional_edges("retrieve", route_after_retrieve, {"quiz": "quiz", "answer": "answer"})
    g.add_edge("answer", "judge")
    g.add_conditional_edges("judge", after_judge, {"rewrite": "rewrite", "end": END})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("offtopic", END)
    g.add_edge("quiz", END)
    return g.compile()


def _valid_quiz_item(d) -> bool:
    return (isinstance(d, dict) and isinstance(d.get("options"), list) and len(d["options"]) == 4
            and isinstance(d.get("correct"), int) and 0 <= d["correct"] <= 3
            and d.get("question") and d.get("explanation"))


def ask(app, question: str, history: list[dict] | None = None) -> dict:
    """Esegue il grafo su una domanda e ritorna lo stato finale."""
    return app.invoke({"question": question, "messages": (history or []) + [{"role": "user", "content": question}]})
