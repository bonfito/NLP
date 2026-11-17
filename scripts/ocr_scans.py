"""OCR (con cache) dei PDF scansionati: Cap2_RabinerJuang.pdf e LinguistGuide2Stats.pdf."""
import sys
from pathlib import Path
sys.path.insert(0, "src")
from nlp_agent.ingest.loaders import ocr_pdf

for name in sys.argv[1:] or ["Cap2_RabinerJuang.pdf", "LinguistGuide2Stats.pdf"]:
    pages = ocr_pdf(Path("data/raw") / name, Path("data/ocr_cache"))
    print(name, len(pages), "pagine,", sum(len(p) for p in pages), "caratteri", flush=True)
