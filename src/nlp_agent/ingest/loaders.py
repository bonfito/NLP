"""Lettura di PDF e HTML in 'unità' (pagina/slide oppure sezione) con testo pulito."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HEADER_PATTERNS = [
    re.compile(r"^\s*Fabio Tamburini\s*\d*\s*$"),
    re.compile(r"^\s*[—–\-_=]{5,}\s*$"),
    re.compile(r"^\s*Outline\s+First\s+Last\s+Prev\s+Next.*$"),
    re.compile(r"^\s*\d{1,3}\s*$"),  # numeri di pagina isolati
]
HTML_JUNK = ("toggle menu", "download article", "go to the original", "a11y-light", "min read")


@dataclass
class Unit:
    """Una pagina/slide (PDF) o una sezione (HTML)."""
    text: str
    page: int | None = None          # 1-based, solo PDF
    heading: str | None = None       # solo HTML
    blocks: list[str] = field(default_factory=list)


@dataclass
class LoadedDoc:
    units: list[Unit]
    needs_ocr: bool = False
    ocr_used: bool = False
    warnings: list[str] = field(default_factory=list)


def clean_text(text: str) -> str:
    text = text.replace("\f", "\n").replace("­", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    lines = []
    for line in text.splitlines():
        if any(p.match(line) for p in HEADER_PATTERNS):
            continue
        lines.append(re.sub(r"[ \t]+", " ", line).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def is_garbled(text: str) -> bool:
    """True se il testo estratto è illeggibile (caratteri di controllo / font senza mappa Unicode)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    ctrl = sum(1 for c in text if (ord(c) < 32 and c not in "\n\t\r\f") or c == "�" or 0xE000 <= ord(c) <= 0xF8FF)
    alpha = sum(1 for c in chars if c.isalpha())
    return ctrl / max(len(text), 1) > 0.02 or (len(text) > 80 and alpha / len(chars) < 0.45)


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def _ocr_page(page, lang: str = "eng", dpi: int = 200) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "p.png"
        page.get_pixmap(dpi=dpi).save(str(png))
        res = subprocess.run(["tesseract", str(png), "stdout", "-l", lang], capture_output=True, text=True)
        return res.stdout


def ocr_pdf(path: Path, cache_dir: Path, lang: str = "eng") -> list[str]:
    """OCR di tutte le pagine, con cache su disco (l'OCR è lento)."""
    import pymupdf

    cache = cache_dir / f"{path.stem}.{lang}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    doc = pymupdf.open(path)
    pages = [_ocr_page(p, lang) for p in doc]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    return pages


def load_pdf(path: Path, kind: str = "article", ocr: bool = True,
             cache_dir: Path = Path("data/ocr_cache")) -> LoadedDoc:
    import pymupdf

    doc = pymupdf.open(path)
    raw_pages, block_pages = [], []
    for page in doc:
        raw_pages.append(page.get_text("text"))
        if kind == "slide":
            block_pages.append([])
        else:
            blocks = [clean_text(b[4]) for b in page.get_text("blocks") if b[6] == 0]
            block_pages.append([b for b in blocks if b])

    n = len(raw_pages)
    empty = sum(1 for t in raw_pages if len(t.strip()) < 20)
    garbled = sum(1 for t in raw_pages if len(t.strip()) >= 20 and is_garbled(t))
    # Nelle slide molte pagine sono solo immagini: normale, niente OCR.
    # In articoli/libri, se la maggioranza delle pagine è vuota o illeggibile serve l'OCR.
    needs_ocr = kind != "slide" and n > 0 and (empty + garbled) / n >= 0.5
    result = LoadedDoc(units=[], needs_ocr=needs_ocr)

    if needs_ocr:
        if ocr and tesseract_available():
            result.ocr_used = True
            result.warnings.append("testo non estraibile: usato OCR (formule e tabelle possono essere imprecise)")
            for i, t in enumerate(ocr_pdf(path, cache_dir), 1):
                t = clean_text(t)
                if t:
                    result.units.append(Unit(text=t, page=i,
                                             blocks=[b.strip() for b in re.split(r"\n\s*\n", t) if b.strip()]))
            return result
        result.warnings.append("testo non estraibile e OCR non disponibile: il file NON sarà indicizzato")
        return result

    for i, (t, blocks) in enumerate(zip(raw_pages, block_pages), 1):
        t = clean_text(t)
        if not t or is_garbled(t):
            continue
        result.units.append(Unit(text=t, page=i, blocks=[b for b in blocks if not is_garbled(b)]))
    if not result.units:
        result.warnings.append("nessun testo estratto")
    return result


def load_html(path: Path) -> LoadedDoc:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "svg", "button", "noscript", "form"]):
        tag.decompose()
    root = soup.find("article") or soup.body or soup

    sections: list[tuple[str, list[str]]] = [("Introduzione", [])]
    last = ""
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre"]):
        text = el.get_text("\n").strip() if el.name == "pre" else re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if not text or (len(text) < 60 and any(j in text.lower() for j in HTML_JUNK)):
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            sections.append((text[:120], []))
        else:
            if text == last:   # blocchi di codice duplicati nella pagina
                continue
            sections[-1][1].append(text)
        last = text

    units = [Unit(text="\n\n".join(b).strip(), heading=h, blocks=b)
             for h, b in sections if len("\n\n".join(b).strip()) >= 40]
    return LoadedDoc(units=units, warnings=[] if units else ["nessun testo estratto"])


def load_file(path: Path, kind: str, ocr: bool = True, cache_dir: Path = Path("data/ocr_cache")) -> LoadedDoc:
    suffix = path.suffix.lower()
    if suffix in (".html", ".htm"):
        return load_html(path)
    if suffix == ".pdf":
        return load_pdf(path, kind=kind, ocr=ocr, cache_dir=cache_dir)
    return LoadedDoc(units=[], warnings=[f"formato non supportato: {suffix}"])
