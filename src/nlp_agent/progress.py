"""Tracciamento dei progressi: esito delle risposte ai quiz, per macroargomento (file JSON locale)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

WEAK_BELOW = 0.7   # sotto il 70% di risposte corrette un macroargomento è "da ripassare"


class ProgressStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data.get("attempts", []) if isinstance(data, dict) else []

    def _save(self, attempts: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"attempts": attempts}, ensure_ascii=False, indent=1), encoding="utf-8")

    def record(self, macro: str, correct: bool, question: str = "", file: str = "") -> None:
        attempts = self._load()
        attempts.append({"ts": datetime.now().isoformat(timespec="seconds"), "macro": macro, "file": file,
                         "question": question, "correct": bool(correct)})
        self._save(attempts)

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def stats(self) -> dict[str, dict]:
        """macroargomento -> {"attempts": n, "correct": c, "accuracy": c/n}."""
        out: dict[str, dict] = {}
        for a in self._load():
            s = out.setdefault(a["macro"], {"attempts": 0, "correct": 0})
            s["attempts"] += 1
            s["correct"] += 1 if a.get("correct") else 0
        for s in out.values():
            s["accuracy"] = s["correct"] / s["attempts"]
        return out

    def totals(self) -> tuple[int, int]:
        st = self.stats().values()
        return sum(s["correct"] for s in st), sum(s["attempts"] for s in st)

    def weakest(self, n: int = 3) -> list[str]:
        """Macroargomenti sotto soglia, dal più debole; a parità, quello con più tentativi."""
        weak = [(m, s) for m, s in self.stats().items() if s["accuracy"] < WEAK_BELOW]
        weak.sort(key=lambda x: (x[1]["accuracy"], -x[1]["attempts"]))
        return [m for m, _ in weak[:n]]

    def summary_text(self) -> str:
        st = self.stats()
        if not st:
            return ("Non ho ancora risultati dei quiz. Fai un quiz (comando `quiz` o interfaccia web) "
                    "e poi ti dirò quali argomenti ripassare.")
        ok, tot = self.totals()
        lines = [f"Hai risposto a {tot} domande di quiz: {ok} corrette ({round(100 * ok / tot)}%).", "",
                 "Per macroargomento:"]
        for m, s in sorted(st.items(), key=lambda x: x[1]["accuracy"]):
            lines.append(f"- {m}: {s['correct']}/{s['attempts']} ({round(100 * s['accuracy'])}%)")
        weak = self.weakest()
        if weak:
            lines += ["", "Da ripassare: " + ", ".join(f"{m} ({round(100 * st[m]['accuracy'])}%)" for m in weak) + ".",
                      f"Puoi scrivere «ripasso di {weak[0]}» oppure «quiz su {weak[0]}»."]
        else:
            lines += ["", "Nessun macroargomento sotto il 70%: continua così, oppure prova un quiz su un nuovo argomento."]
        return "\n".join(lines)
