"""Stato condiviso del grafo (è il 'contratto' tra i nodi)."""
from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from .store import Passage


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], add]   # cronologia {"role": "user"|"assistant", "content": str}
    question: str                          # ultima richiesta dell'utente
    route: str                             # theory | quiz | review | progress | offtopic
    query: str                             # query di retrieval (può essere riformulata dal judge)
    topic: str                             # argomento per quiz/ripasso
    passages: list[Passage]
    answer: str
    verdict: dict                          # {"grounded": bool, "issues": str, "new_query": str}
    revisions: int                         # cicli di revisione già fatti
    corrections: int                       # quante volte il judge ha trovato un problema (per la valutazione)
    sources: list[dict]
    quiz_items: list[dict]                 # quiz strutturato: question, options, correct, explanation, macro, source
