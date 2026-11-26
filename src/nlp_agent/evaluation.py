"""Valutazione: agente (RAG + judge) contro lo stesso LLM senza retrieval.

Metriche per domanda:
  - correct: risposta corretta rispetto alla risposta attesa (giudicata dall'LLM, 0/1)
  - source_file_ok / source_macro_ok: tra le fonti indicate c'è un file / un macroargomento atteso (solo agente)
  - corrections: quante volte il judge ha trovato un problema e rilanciato il retrieval (solo agente)
  - route_ok: per le domande fuori tema, l'agente ha rifiutato
  - opinion_ok: per le domande con tesi del docente, la risposta la attribuisce al docente
Attenzione: l'LLM-giudice è lo stesso modello che risponde; controllate a mano un campione.
"""
from __future__ import annotations

import json
from statistics import mean

from .graph import ask
from .llm import LLM, extract_json

GRADER = """Sei un correttore. Ti vengono dati una DOMANDA, la RISPOSTA ATTESA e la RISPOSTA DA VALUTARE.
La risposta è corretta se contiene i punti essenziali della risposta attesa senza contraddirli.
Rispondi SOLO con JSON: {"correct": true|false, "reason": "breve motivazione"}"""

BASELINE_SYSTEM = ("Sei un assistente per il corso di Natural Language Processing. Rispondi in italiano, "
                   "mantenendo i termini tecnici in inglese.")


def grade(llm: LLM, question: str, expected: str, answer: str) -> bool:
    out = extract_json(llm.complete(GRADER, f"DOMANDA: {question}\n\nRISPOSTA ATTESA: {expected}\n\nRISPOSTA DA VALUTARE:\n{answer}"))
    return bool(out.get("correct")) if isinstance(out, dict) else False


def run_eval(app, llm: LLM, questions: list[dict], baseline: bool = True) -> dict:
    rows = []
    for q in questions:
        row = {"id": q["id"], "question": q["question"], "route_expected": q.get("route", "theory")}
        state = ask(app, q["question"])
        row.update(route=state["route"], answer=state["answer"], corrections=state.get("corrections", 0),
                   sources=state.get("sources", []))
        if row["route_expected"] == "offtopic":
            row["route_ok"] = state["route"] == "offtopic"
        else:
            row["correct"] = grade(llm, q["question"], q["expected_answer"], state["answer"].split("\n\nFonti:")[0])
            files = {s["file"] for s in row["sources"]}
            row["source_file_ok"] = bool(files & set(q.get("expected_files", [])))
            row["source_macro_ok"] = any(s["macro"] == q.get("expected_macro") for s in row["sources"])
            if q.get("expect_opinion_flag"):
                row["opinion_ok"] = "docente" in state["answer"].lower()
            if baseline:
                b = llm.complete(BASELINE_SYSTEM, q["question"])
                row["baseline_answer"] = b
                row["baseline_correct"] = grade(llm, q["question"], q["expected_answer"], b)
        rows.append(row)
    return {"rows": rows, "summary": summarize(rows)}


def _rate(rows, key):
    vals = [r[key] for r in rows if key in r]
    return round(mean(vals), 3) if vals else None


def summarize(rows: list[dict]) -> dict:
    theory = [r for r in rows if r["route_expected"] != "offtopic"]
    return {
        "n_theory": len(theory),
        "n_offtopic": len(rows) - len(theory),
        "agent_correct": _rate(theory, "correct"),
        "baseline_correct": _rate(theory, "baseline_correct"),
        "source_file_accuracy": _rate(theory, "source_file_ok"),
        "source_macro_accuracy": _rate(theory, "source_macro_ok"),
        "opinion_flag_rate": _rate(theory, "opinion_ok"),
        "offtopic_refusal": _rate([r for r in rows if "route_ok" in r], "route_ok"),
        "questions_with_judge_correction": sum(1 for r in theory if r["corrections"] > 0),
        "total_judge_corrections": sum(r["corrections"] for r in theory),
    }


def to_markdown(summary: dict) -> str:
    pct = lambda v: "n/d" if v is None else f"{v * 100:.0f}%"
    return "\n".join([
        "| Metrica | Valore |", "|---|---|",
        f"| Domande di teoria / fuori tema | {summary['n_theory']} / {summary['n_offtopic']} |",
        f"| Correttezza agente (RAG+judge) | {pct(summary['agent_correct'])} |",
        f"| Correttezza baseline (stesso LLM, senza retrieval) | {pct(summary['baseline_correct'])} |",
        f"| Fonte: file corretto | {pct(summary['source_file_accuracy'])} |",
        f"| Fonte: macroargomento corretto | {pct(summary['source_macro_accuracy'])} |",
        f"| Tesi del docente attribuite al docente | {pct(summary['opinion_flag_rate'])} |",
        f"| Rifiuto domande fuori tema | {pct(summary['offtopic_refusal'])} |",
        f"| Domande in cui il judge è intervenuto | {summary['questions_with_judge_correction']} |",
    ])


def select_questions(questions: list[dict], limit: int | None) -> list[dict]:
    """Con --limit N prende domande a rotazione dai vari gruppi (prefisso dell'id: emb, dl, off, en, ...),
    così un test ridotto include argomenti diversi, fuori tema e tesi del docente, non solo le prime N."""
    if not limit or limit >= len(questions):
        return questions
    groups: dict[str, list[dict]] = {}
    for q in questions:
        groups.setdefault(q["id"].split("-")[0], []).append(q)
    # i gruppi speciali (fuori tema, tesi del docente) vengono per primi nella rotazione
    order = sorted(groups, key=lambda g: (0 if g == "off" or any(x.get("expect_opinion_flag") for x in groups[g]) else 1))
    out: list[dict] = []
    while len(out) < limit:
        added = False
        for g in order:
            if groups[g] and len(out) < limit:
                out.append(groups[g].pop(0))
                added = True
        if not added:
            break
    return out
