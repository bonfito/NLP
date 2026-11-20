"""Interfaccia minima verso l'LLM: complete(system, user) -> str. Provider: ollama | anthropic | openai."""
from __future__ import annotations

import json
import re
from typing import Protocol

from .config import Settings


class LLM(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class LangChainLLM:
    def __init__(self, chat_model):
        self.chat = chat_model

    def complete(self, system: str, user: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        out = self.chat.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = out.content
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return content


def get_llm(s: Settings) -> LLM:
    p = s.llm_provider.lower()
    if p == "ollama":
        from langchain_ollama import ChatOllama

        return LangChainLLM(ChatOllama(model=s.llm_model, temperature=0))
    if p == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return LangChainLLM(ChatAnthropic(model=s.llm_model, temperature=0))
    if p == "openai":
        # Vale anche per servizi gratuiti compatibili OpenAI (Groq, OpenRouter, ...): basta OPENAI_BASE_URL e OPENAI_API_KEY.
        import os

        from langchain_openai import ChatOpenAI

        kwargs = {"base_url": os.environ["OPENAI_BASE_URL"]} if os.environ.get("OPENAI_BASE_URL") else {}
        return LangChainLLM(ChatOpenAI(model=s.llm_model, temperature=0, max_retries=6, **kwargs))
    raise ValueError(f"LLM_PROVIDER sconosciuto: {s.llm_provider}")


def extract_json(text: str):
    """Estrae il primo oggetto/array JSON da una risposta (anche dentro ```json ... ```). None se non valido.
    Si parte dall'apertura che compare per prima, così una lista di un solo oggetto resta una lista."""
    text = re.sub(r"```(?:json)?", "", text)
    candidates = sorted((text.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if text.find(o) != -1)
    for start, _, closer in candidates:
        end = text.rfind(closer)
        if end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None
