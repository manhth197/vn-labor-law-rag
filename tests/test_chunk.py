from __future__ import annotations

from tests.conftest import parsed, word_count
from vnlaw_rag.ingest.chunk import build_chunks, chunk_appendix, chunk_article
from vnlaw_rag.ingest.models import Article, Clause, Point


def make_article(clauses: list[Clause], intro: str = "Lời dẫn:") -> Article:
    return Article(
        doc_short="BLLD2019",
        doc_number="45/2019/QH14",
        article_no=7,
        article_title="Tiêu đề",
        chapter_seq=1,
        chapter_no="I",
        chapter_title="CHƯƠNG MỘT",
        intro=intro,
        clauses=clauses,
        source_url="https://example.invalid/doc",
    )


def words(n: int) -> str:
    return " ".join(["từ"] * n)


def test_short_article_is_one_chunk_with_context_header():
    article = make_article([Clause(no=1, text="Một."), Clause(no=2, text="Hai.")])
    (chunk,) = chunk_article(article, word_count, max_tokens=100)
    assert chunk.chunk_id == "BLLD2019-D7"
    assert chunk.clause_nos == []
    assert chunk.text.split("\n") == [
        "Bộ luật Lao động 2019 (45/2019/QH14) - Chương I. CHƯƠNG MỘT",
        "Điều 7. Tiêu đề",
        "Lời dẫn:",
        "1. Một.",
        "2. Hai.",
    ]
    assert chunk.n_tokens == word_count(chunk.text)


def test_long_article_is_split_at_clauses_and_repeats_the_lead_in():
    article = make_article([Clause(no=i, text=words(30)) for i in range(1, 6)])
    chunks = chunk_article(article, word_count, max_tokens=90)
    ids = [c.chunk_id for c in chunks]
    assert ids == ["BLLD2019-D7-K1-2", "BLLD2019-D7-K3-4", "BLLD2019-D7-K5"]
    assert [c.clause_nos for c in chunks] == [[1, 2], [3, 4], [5]]
    for c in chunks:
        assert c.n_tokens <= 90
        lines = c.text.split("\n")
        assert lines[1].startswith("Điều 7. Tiêu đề (khoản ")
        assert lines[2] == "Lời dẫn:"


def test_clause_too_long_on_its_own_is_split_at_points():
    big = Clause(no=2, text="Gồm:", points=[Point(letter=x, text=words(40)) for x in "abcd"])
    article = make_article([Clause(no=1, text="Ngắn."), big, Clause(no=3, text="Ngắn.")])
    chunks = chunk_article(article, word_count, max_tokens=110)
    ids = [c.chunk_id for c in chunks]
    assert ids == [
        "BLLD2019-D7-K1",
        "BLLD2019-D7-K2-p1",
        "BLLD2019-D7-K2-p2",
        "BLLD2019-D7-K3",
    ]
    for c in chunks[1:3]:
        assert "\n2. Gồm:\n" in c.text  # the clause lead-in is repeated in each piece
        assert c.n_tokens <= 110


def test_appendix_chunk():
    appendix = parsed("ND145").appendices[1]
    chunk = chunk_appendix(appendix, word_count)
    assert chunk.chunk_id == "ND145-PLII"
    assert chunk.unit == "appendix"
    assert chunk.article_no is None
    assert chunk.text.startswith(
        "Nghị định 145/2020/NĐ-CP (145/2020/NĐ-CP)\nPhụ lục II. DANH MỤC CÔNG VIỆC"
    )


def test_real_corpus_chunks_cover_every_article_within_budget():
    docs = [parsed("BLLD2019"), parsed("ND145")]
    articles = [a for d in docs for a in d.articles]
    appendices = [a for d in docs for a in d.appendices]
    chunks = build_chunks(articles, appendices, word_count, max_tokens=1200)
    covered = {(c.doc_short, c.article_no) for c in chunks if c.unit == "article"}
    assert covered == {(a.doc_short, a.article_no) for a in articles}
    assert all(c.n_tokens <= 1200 for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
