"""Mappa dei materiali del corso: file -> parte, macroargomento, tipo."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import yaml


def norm_name(name: str) -> str:
    """Normalizza un nome file: ignora accenti, maiuscole, ' (1)' e artefatti dello zip (#U00e8)."""
    name = re.sub(r"#U([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), name)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"\s*\(\d+\)(?=\.[^.]+$)", "", name)
    return name.lower().strip()


@dataclass(frozen=True)
class CourseFile:
    part: str
    macro: str
    kind: str          # slide | article | tutorial | book
    stance: str = ""   # "professor_opinion" oppure ""
    verified: bool = True


class CourseMap:
    def __init__(self, files: dict[str, CourseFile], exclude: set[str]):
        self._files = files
        self._exclude = exclude

    @classmethod
    def load(cls, path: str | Path) -> "CourseMap":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        files = {
            norm_name(name): CourseFile(part=v["part"], macro=v["macro"], kind=v["kind"],
                                        stance=v.get("stance", ""), verified=v.get("verified", True))
            for name, v in data["files"].items()
        }
        return cls(files, {norm_name(n) for n in data.get("exclude", [])})

    def is_excluded(self, filename: str) -> bool:
        return norm_name(filename) in self._exclude

    def lookup(self, filename: str) -> CourseFile | None:
        return self._files.get(norm_name(filename))

    @property
    def macros(self) -> list[str]:
        return sorted({f.macro for f in self._files.values()})
