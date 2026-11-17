from nlp_agent.course_map import CourseFile
from nlp_agent.ingest.chunking import chunk_document
from nlp_agent.ingest.loaders import LoadedDoc, Unit, clean_text, is_garbled, load_html

SLIDE = CourseFile(part="Part I", macro="Embeddings", kind="slide")
ART = CourseFile(part="Extra", macro="RAG", kind="article")


def test_clean_text_removes_slide_headers():
    t = "Fabio Tamburini 3\n——————————————————\nTitolo\n 12 \ntesto  con   spazi"
    assert clean_text(t) == "Titolo\ntesto con spazi"


def test_garbled_detection():
    assert is_garbled("\x02\x01\x04\x03 \x05\x07\x06 \b\x04" * 20)
    assert not is_garbled("This is a normal sentence about embeddings and vectors in NLP.")


def test_slides_are_merged_and_locator_is_range():
    doc = LoadedDoc(units=[Unit(text="Slide uno con un po di testo utile " * 3, page=i) for i in range(1, 6)])
    chunks = chunk_document("X.pdf", doc, SLIDE)
    assert len(chunks) < 5
    assert chunks[0].metadata["locator"].startswith("slide 1")
    assert chunks[0].metadata["macro"] == "Embeddings"


def test_article_chunks_use_page_locator_and_respect_max():
    blocks = [("Paragrafo numero %d. " % i) * 20 for i in range(30)]
    doc = LoadedDoc(units=[Unit(text="\n\n".join(blocks), page=3, blocks=blocks)])
    chunks = chunk_document("A.pdf", doc, ART)
    assert len(chunks) > 3
    assert all(len(c.text) <= 1800 for c in chunks)
    assert chunks[0].metadata["locator"] == "p. 3"


def test_html_sections(tmp_path):
    p = tmp_path / "t.html"
    p.write_text("<html><body><article><h2>Concept 1: State</h2>"
                 "<p>State is the shared notebook that every step can read and write.</p>"
                 "<h2>Concept 2: Nodes</h2><p>A node is just a Python function that takes state in.</p>"
                 "<pre>def f(state): return {}\n</pre><script>x=1</script></article></body></html>")
    doc = load_html(p)
    heads = [u.heading for u in doc.units]
    assert "Concept 1: State" in heads and "Concept 2: Nodes" in heads
    assert "x=1" not in " ".join(u.text for u in doc.units)
    chunks = chunk_document("t.html", doc, CourseFile("Part III", "Agenti", "tutorial"))
    assert chunks[0].metadata["locator"].startswith("sezione")
