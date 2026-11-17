"""Configurazione letta da variabili d'ambiente (vedi .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    env = Path(".env")
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.split("#")[0].strip())


@dataclass
class Settings:
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:7b-instruct"
    embedding_model: str = "intfloat/multilingual-e5-small"
    raw_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")
    course_map: Path = Path("data/course_map.yaml")
    top_k: int = 6
    max_revisions: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        g = os.environ.get
        return cls(
            llm_provider=g("LLM_PROVIDER", cls.llm_provider),
            llm_model=g("LLM_MODEL", cls.llm_model),
            embedding_model=g("EMBEDDING_MODEL", cls.embedding_model),
            raw_dir=Path(g("RAW_DIR", "data/raw")),
            index_dir=Path(g("INDEX_DIR", "data/index")),
            course_map=Path(g("COURSE_MAP", "data/course_map.yaml")),
            top_k=int(g("TOP_K", cls.top_k)),
            max_revisions=int(g("MAX_REVISIONS", cls.max_revisions)),
        )
