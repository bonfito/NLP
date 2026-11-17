from nlp_agent.course_map import CourseMap, norm_name


def test_norm_name_zip_artifacts():
    assert norm_name("Milli#U00e8re2024.pdf") == "milliere2024.pdf"
    assert norm_name("GrammarsAndParsing (1).pdf") == "grammarsandparsing.pdf"


def test_map_covers_course_files():
    cm = CourseMap.load("data/course_map.yaml")
    assert cm.lookup("Milli#U00e8re2024.pdf").macro == "DNN e cognizione"
    assert cm.lookup("GrammarsAndParsing (1).pdf").part == "Part II"
    assert cm.is_excluded("SLP_ed3book_aug25.pdf")
    assert cm.lookup("sconosciuto.pdf") is None
    assert cm.lookup("SOTA4LLMs2026.pdf").stance == "professor_opinion"
