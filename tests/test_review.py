from __future__ import annotations

import shutil
from collections import Counter

import pytest

from tests.conftest import REPO_ROOT, parsed
from vnlaw_rag.eval.golden import Ref, load_golden, unit_texts
from vnlaw_rag.eval.review import (
    NO_REFS,
    export_review,
    format_refs,
    import_review,
    parse_refs,
    stratified_sample,
)

GOLDEN = REPO_ROOT / "data" / "eval" / "golden.jsonl"


def test_stratified_sample_keeps_type_shares_and_is_reproducible():
    items = load_golden(GOLDEN)
    sample = stratified_sample(items, 25, seed=42)
    assert len(sample) == 25
    counts = Counter(i.type for i in sample)
    full = Counter(i.type for i in items)
    for t in full:  # within one of the exact proportional share
        assert abs(counts[t] - 25 * full[t] / len(items)) < 1
    assert counts["unanswerable"] >= 1
    assert [i.id for i in stratified_sample(items, 25, seed=42)] == [i.id for i in sample]


def test_refs_round_trip_through_the_cell_format():
    refs = [Ref(doc="BLLD2019", article=25), Ref(doc="ND145", appendix="II")]
    cell = format_refs(refs, {("BLLD2019", "article", "25"): "Thời gian thử việc"})
    assert cell.splitlines()[0] == "BLLĐ 2019 - Điều 25: Thời gian thử việc"
    assert [r.key() for r in parse_refs(cell)] == [r.key() for r in refs]
    assert format_refs([], {}) == NO_REFS
    assert parse_refs("nđ 145/2020 Điều 7") == [Ref(doc="ND145", article=7)]


@pytest.fixture
def workbook(tmp_path):
    pytest.importorskip("openpyxl")
    docs = [parsed("BLLD2019"), parsed("ND145")]
    articles = [a for d in docs for a in d.articles]
    appendices = [a for d in docs for a in d.appendices]
    titles = {(a.doc_short, "article", str(a.article_no)): a.article_title for a in articles}
    golden = tmp_path / "golden.jsonl"
    shutil.copy(GOLDEN, golden)
    out = tmp_path / "review.xlsx"
    sample = export_review(golden, unit_texts(articles, appendices), titles, out, n=25)
    return golden, out, sample


def test_export_writes_headers_excerpts_and_dropdown(workbook):
    from openpyxl import load_workbook

    _, out, sample = workbook
    ws = load_workbook(out)["Duyệt"]
    assert ws["A1"].value == "id" and ws["G1"].value == "Đánh giá"
    assert ws.max_row == len(sample) + 1
    first = next(i for i in sample if i.expected)
    row = next(r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] == first.id)
    assert row[2] == first.question
    assert len(row[4].split("\n\n")[0]) <= 301  # 300 characters + ellipsis
    assert ws.data_validations.dataValidation[0].formula1 == '"Đúng,Cần sửa,Bỏ"'


def test_import_applies_verdicts_and_edits(workbook, tmp_path):
    from openpyxl import load_workbook

    golden, out, sample = workbook
    wb = load_workbook(out)
    ws = wb["Duyệt"]
    rows = {ws.cell(row=r, column=1).value: r for r in range(2, ws.max_row + 1)}
    answerable = [i for i in sample if i.expected]
    ok, fixed, noted, dropped = answerable[:4]
    ws.cell(rows[ok.id], 7).value = "Đúng"
    ws.cell(rows[fixed.id], 7).value = "Cần sửa"
    ws.cell(rows[fixed.id], 3).value = "Câu hỏi đã sửa?"
    ws.cell(rows[fixed.id], 4).value = "BLLĐ 2019 - Điều 26"
    ws.cell(rows[noted.id], 7).value = "Cần sửa"
    ws.cell(rows[noted.id], 8).value = "Sai Điều"
    ws.cell(rows[dropped.id], 7).value = "Bỏ"
    wb.save(out)

    removed = tmp_path / "removed.jsonl"
    summary = import_review(out, golden, removed)
    items = {i.id: i for i in load_golden(golden)}
    assert summary.counts == Counter(ok=1, fixed=1, needs_fix=1, removed=1, skipped=21)
    assert items[ok.id].reviewed and items[ok.id].review_status == "ok"
    assert items[fixed.id].question == "Câu hỏi đã sửa?"
    assert [r.key() for r in items[fixed.id].expected] == [("BLLD2019", "article", "26")]
    assert items[fixed.id].reviewed and items[fixed.id].review_status == "fixed"
    assert not items[noted.id].reviewed and items[noted.id].review_note == "Sai Điều"
    assert dropped.id not in items
    assert dropped.id in removed.read_text(encoding="utf-8")
