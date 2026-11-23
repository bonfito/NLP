"""Interfaccia minima verso l'LLM: complete(system, user) -> str. Provider: ollama | anthropic | openai."""
from __future__ import annotations

import json
import re
from typing import Protocol

from .config import Settings


class LLM(Protocol):
    def complete(self, system: str, user: str, json_out: bool = False) -> str: ...


class LangChainLLM:
    def __init__(self, chat_model, json_model=None):
        self.chat = chat_model
        self.json_chat = json_model or chat_model   # modello "breve" per le uscite JSON (router, judge)

    def complete(self, system: str, user: str, json_out: bool = False) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        import os
        import sys
        import time

        t0 = time.perf_counter()
        out = (self.json_chat if json_out else self.chat).invoke([SystemMessage(content=system), HumanMessage(content=user)])
        if os.environ.get("NLP_DEBUG"):
            dt = time.perf_counter() - t0
            print(f"[debug] chiamata LLM: {dt:.1f}s, prompt {len(system) + len(user)} caratteri, "
                  f"risposta {len(str(out.content))} caratteri", file=sys.stderr, flush=True)
        content = out.content
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


def get_llm(s: Settings) -> LLM:
    p = s.llm_provider.lower()
    if p == "ollama":
        from langchain_ollama import ChatOllama

        import os

        # I modelli "thinking" (Qwen3.x) sul solo processore spendono minuti a ragionare e a volte restituiscono
        # contenuto vuoto: di default il ragionamento è spento. LLM_THINKING=true per riattivarlo.
        num_ctx = {"num_ctx": int(os.environ["LLM_NUM_CTX"])} if os.environ.get("LLM_NUM_CTX") else {}
        thinking = os.environ.get("LLM_THINKING", "false").strip().lower() in ("1", "true", "yes")
        common = dict(model=s.llm_model, temperature=0, reasoning=thinking, **num_ctx)
        # Su CPU il costo è quasi tutto nei token generati: risposta con tetto, JSON di router/judge forzato e corto.
        return LangChainLLM(ChatOllama(num_predict=900, **common),
                            ChatOllama(num_predict=300, format="json", **common))
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
