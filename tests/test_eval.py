from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import REPO_ROOT, parsed
from vnlaw_rag.eval.golden import GoldenItem, Ref, load_golden, unit_texts, validate_golden
from vnlaw_rag.eval.retrieval import score_question, summarize
from vnlaw_rag.retrieval.hybrid import RetrievedChunk

GOLDEN = REPO_ROOT / "data" / "eval" / "golden.jsonl"


@pytest.fixture(scope="module")
def texts():
    docs = [parsed("BLLD2019"), parsed("ND145")]
    return unit_texts(
        [a for d in docs for a in d.articles], [a for d in docs for a in d.appendices]
    )


def test_golden_set_is_consistent_with_the_corpus(texts):
    items = load_golden(GOLDEN)
    assert 60 <= len(items) <= 80
    assert {i.type for i in items} == {"lookup", "paraphrase", "multi", "numeric", "unanswerable"}
    assert validate_golden(items, texts) == []


def test_validation_catches_a_wrong_article(texts):
    item = GoldenItem(
        id="x",
        question="Thử việc tối đa bao lâu?",
        type="numeric",
        expected=[Ref(doc="BLLD2019", article=26)],  # the quote is in Điều 25
        evidence=["Không quá 60 ngày"],
        reference_answer="",
    )
    (problem,) = validate_golden([item], texts)
    assert "evidence not found" in problem


def chunk(doc: str, article: int | None = None, appendix: str | None = None) -> RetrievedChunk:
    unit = "appendix" if appendix else "article"
    row = {"doc_short": doc, "unit": unit, "article_no": article, "appendix_no": appendix}
    return RetrievedChunk(chunk_id=f"{doc}-{article or appendix}", score=0.0, vector_score=0.5,
                          row=row)  # fmt: skip


def item(type_: str, *refs: Ref) -> GoldenItem:
    return GoldenItem(id=type_, question="q", type=type_, expected=list(refs), reference_answer="")


def test_score_question_ranks_and_recall():
    retrieved = [chunk("BLLD2019", 1), chunk("ND145", appendix="II"), chunk("BLLD2019", 25)]
    multi = item("multi", Ref(doc="BLLD2019", article=25), Ref(doc="ND145", appendix="II"))
    r = score_question(multi, retrieved)
    assert r.first_relevant_rank == 2
    assert r.recall_at_10 == 1.0
    assert r.hit(1) is False and r.hit(5) is True
    assert r.reciprocal_rank == 0.5


def test_summarize_excludes_unanswerable(tmp_path: Path):
    hit = score_question(item("lookup", Ref(doc="BLLD2019", article=1)), [chunk("BLLD2019", 1)])
    miss = score_question(item("numeric", Ref(doc="BLLD2019", article=9)), [chunk("BLLD2019", 1)])
    none = score_question(item("unanswerable"), [chunk("BLLD2019", 1)])
    summary = summarize([hit, miss, none])
    assert summary["all"]["n"] == 2
    assert summary["all"]["hit@1"] == 0.5
    assert summary["lookup"]["mrr@10"] == 1.0
    assert "unanswerable" not in summary


def test_default_label_matches_configs():
    from vnlaw_rag.config import Settings
    from vnlaw_rag.eval.report import CONFIGS, default_label

    labels = {c[0] for c in CONFIGS}
    assert default_label(Settings(retrieval_mode="vector")) == "vector"
    hybrid = Settings(retrieval_mode="hybrid", keyword_scoring="idf", keyword_stopwords=True)
    assert default_label(hybrid) == "hybrid-bigram-idf-stop"
    assert default_label(hybrid) in labels


def test_write_report_replaces_only_the_retrieval_section(tmp_path, monkeypatch):
    from vnlaw_rag.eval import report

    monkeypatch.setattr(
        report, "render_markdown", lambda run: f"{report.START}\nNEW\n{report.END}\n"
    )
    md = tmp_path / "EVAL.md"
    md.write_text(f"# Evaluation\n\n{report.START}\nOLD\n{report.END}\n\n## Answers\nkeep\n")
    report.write_report({"items": {}}, md, tmp_path / "r.json")
    text = md.read_text(encoding="utf-8")
    assert "NEW" in text and "OLD" not in text
    assert text.endswith("## Answers\nkeep\n")
