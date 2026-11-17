"""Controllo rapido dell'ingestion senza embeddings: quanti chunk per file, avvisi, file non mappati."""
import sys
from pathlib import Path
sys.path.insert(0, "src")
from nlp_agent.course_map import CourseMap
from nlp_agent.ingest.chunking import chunk_document
from nlp_agent.ingest.loaders import load_file

raw = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), "data/raw"))
cmap = CourseMap.load("data/course_map.yaml")
tot = 0
for p in sorted(raw.iterdir()):
    if p.suffix.lower() not in (".pdf", ".html"):
        continue
    if cmap.is_excluded(p.name):
        print(f"{'ESCLUSO':>8} | {p.name}"); continue
    info = cmap.lookup(p.name)
    if not info:
        print(f"{'NON MAPP.':>8} | {p.name}"); continue
    doc = load_file(p, info.kind, ocr="--ocr" in sys.argv, cache_dir=Path("data/ocr_cache"))
    ch = chunk_document(p.name, doc, info)
    tot += len(ch)
    w = "; ".join(doc.warnings)
    print(f"{len(ch):>8} | {p.name:<48} {info.kind:<8} {w}")
print("TOTALE chunk:", tot)
