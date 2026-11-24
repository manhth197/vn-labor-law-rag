"""Answer eval: numeric facts, citation scoring, stop rules, resume, report."""

from __future__ import annotations

import json

from vnlaw_rag.config import REPO_ROOT
from vnlaw_rag.eval.answers import (
    JudgeScore,
    RequestLog,
    facts,
    item_key_fact,
    key_fact,
    load_done,
    merge_rerun,
    render_answers_markdown,
    render_current_markdown,
    rescore_numeric,
    run_answer_eval,
    score,
    summarize_answers,
)
from vnlaw_rag.eval.golden import GoldenItem, Ref, load_golden
from vnlaw_rag.llm.base import LLMError, RateLimited, StructuredResult


def test_facts_normalize_digits_billions_and_number_words():
    assert facts("Không quá 06 tháng; ký quỹ 2.000.000.000 đồng") == [
        ("6", "tháng"),
        ("2000000000", "đồng"),
    ]
    assert facts("ký quỹ 2 tỷ đồng, nghỉ năm ngày, mười hai tháng") == [
        ("2000000000", "đồng"), ("5", "ngày"), ("12", "tháng"),
    ]  # fmt: skip
    assert facts("Điều 25 khoản 2") == []  # law references carry no unit
    assert facts("hệ số 0,7 lần") == [("0.7", "lần")]


def test_key_fact_is_the_first_figure_of_the_reference():
    assert key_fact("Không quá 60 ngày (khoản 2 Điều 25 BLLĐ 2019).") == ("60", "ngày")
    assert key_fact("Từ năm 2021 là đủ 60 tuổi 03 tháng") == ("60", "tuổi")
    assert key_fact("Ít nhất 85% mức lương") == ("85", "%")
    assert key_fact("Có, nếu thỏa thuận") is None


def item(type_="numeric", expected=(("BLLD2019", 25),), answer="Không quá 60 ngày.", qid="q1"):
    return GoldenItem(
        id=qid, question="q", type=type_, reference_answer=answer, reviewed=True,
        expected=[Ref(doc=d, article=a) for d, a in expected],
    )  # fmt: skip


def source(chunk_id, article):
    return {"chunk_id": chunk_id, "doc_short": "BLLD2019", "unit": "article",
            "article_no": article, "content": "..."}  # fmt: skip


def test_score_citations_and_numeric_match():
    out = {
        "status": "answered",
        "answer": "Thời gian thử việc không quá 60 ngày [BLLD2019-D25].",
        "citations": ["BLLD2019-D25", "BLLD2019-D24"],
        "sources": [source("BLLD2019-D25", 25), source("BLLD2019-D24", 24)],
    }
    row = score(item(), out)
    assert row["precision"] == 0.5 and row["recall"] == 1.0
    assert row["numeric_match"] is True and row["key_fact"] == "60 ngày"
    assert row["refusal_ok"] is True


def test_score_refusals():
    refused = {"status": "not_found", "answer": "Không tìm thấy", "citations": [], "sources": []}
    assert score(item(type_="unanswerable", expected=()), refused)["refusal_ok"] is True
    wrong = score(item(), refused)
    assert wrong["refusal_ok"] is False and wrong["numeric_match"] is False
    answered = {**refused, "status": "answered"}
    assert score(item(type_="unanswerable", expected=()), answered)["refusal_ok"] is False


class FakeGraph:
    def __init__(self, outcomes, requests):
        self.outcomes = list(outcomes)
        self.requests = requests

    def invoke(self, state):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            self.requests("gen", False)
            raise outcome
        self.requests("gen", True)
        return outcome


class FakeJudge:
    def __init__(self, requests):
        self.requests = requests
        self.calls = 0

    def generate_structured(self, **kwargs):
        self.calls += 1
        self.requests("judge", True)
        return StructuredResult(JudgeScore(correctness=4, faithfulness=5, rationale="ok"), "m")


GOOD = {
    "status": "answered", "answer": "Không quá 60 ngày [BLLD2019-D25].",
    "citations": ["BLLD2019-D25"], "sources": [source("BLLD2019-D25", 25)],
}  # fmt: skip


def test_run_scores_and_judges_each_answered_question(tmp_path):
    path = tmp_path / "results.jsonl"
    requests = RequestLog()
    judge = FakeJudge(requests)
    items = [item(qid="q1"), item(qid="q2")]
    info = run_answer_eval(
        items, FakeGraph([GOOD, GOOD], requests), judge, path, requests,
        pause_seconds=0, sleep=lambda s: None,
    )  # fmt: skip
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["q1", "q2"]
    assert all(r["judge"]["correctness"] == 4 for r in rows) and judge.calls == 2
    assert info == {"stopped": None, "requests": 4, "failures": 0}


def test_stop_rule_after_20_requests_with_more_than_20_percent_failures(tmp_path):
    requests = RequestLog()
    outcomes = [LLMError("boom")] * 5 + [GOOD] * 30
    items = [item(qid=f"q{i}") for i in range(30)]
    info = run_answer_eval(items, FakeGraph(outcomes, requests), FakeJudge(requests),
                           tmp_path / "r.jsonl", requests, pause_seconds=0,
                           sleep=lambda s: None)  # fmt: skip
    assert info["stopped"] == "more than 20% of the first 20 requests failed"
    assert info["requests"] < 30


def test_rate_limit_stops_with_partial_results_and_resume_skips_done(tmp_path):
    path = tmp_path / "r.jsonl"
    requests = RequestLog()
    items = [item(qid="a"), item(qid="b"), item(qid="c")]
    info = run_answer_eval(items, FakeGraph([GOOD, RateLimited("quota")], requests),
                           FakeJudge(requests), path, requests, pause_seconds=0,
                           sleep=lambda s: None)  # fmt: skip
    assert info["stopped"].startswith("rate limited at b")
    assert [json.loads(x)["id"] for x in path.read_text("utf-8").splitlines()] == ["a"]
    requests2 = RequestLog()
    run_answer_eval(items, FakeGraph([GOOD, GOOD], requests2), FakeJudge(requests2), path,
                    requests2, pause_seconds=0, sleep=lambda s: None)  # fmt: skip
    assert [json.loads(x)["id"] for x in path.read_text("utf-8").splitlines()] == ["a", "b", "c"]


def test_summary_and_report():
    rows = [
        score(item(qid="n1"), GOOD) | {"judge": {"correctness": 5, "faithfulness": 5}},
        score(item(qid="n2"), {**GOOD, "answer": "Không quá 30 ngày [BLLD2019-D25]."}),
        score(item(type_="unanswerable", expected=(), qid="u1"),
              {"status": "not_found", "answer": "", "citations": [], "sources": []}),
    ]  # fmt: skip
    summary = summarize_answers(rows)
    assert summary["numeric_exact_match"] == 0.5
    assert summary["correct_refusal_rate"] == 1.0
    assert summary["judge_correctness"] == 5
    md = render_answers_markdown(
        {"date": "2026-11-24", "git": "abc", "model": "m", "judge_model": "m", "pause_s": 4,
         "golden_n": 77, "requests": 10, "failures": 1, "stopped": None}, rows,
    )  # fmt: skip
    assert "same model as the generator" in md
    assert "`n2` (numeric): key figure '60 ngày' missing" in md
    assert "failed: 1 (10.0%)" in md


def test_explicit_key_fact_overrides_the_first_figure():
    per_week = item(answer="Không quá 8 giờ/ngày và không quá 48 giờ trong 01 tuần.")
    assert item_key_fact(per_week) == ("8", "giờ")
    per_week.key_fact = "48 giờ"
    assert item_key_fact(per_week) == ("48", "giờ")
    per_week.key_fact = "không có số"  # unusable annotation: fall back to the first figure
    assert item_key_fact(per_week) == ("8", "giờ")


def test_rescore_numeric_uses_the_annotation_without_touching_other_fields():
    golden = item(answer="Không quá 8 giờ/ngày và 48 giờ/tuần.")
    out = {**GOOD, "answer": "Tối đa 48 giờ trong một tuần [BLLD2019-D25]."}
    row = score(golden, out)
    assert row["numeric_match"] is False and row["key_fact"] == "8 giờ"
    before = {k: v for k, v in row.items() if k not in ("numeric_match", "key_fact")}
    golden.key_fact = "48 giờ"
    error_row = {"id": "q1", "status": "error", "numeric_match": None}
    rescore_numeric([row, error_row], {"q1": golden})
    assert row["numeric_match"] is True and row["key_fact"] == "48 giờ"
    assert {k: v for k, v in row.items() if k not in ("numeric_match", "key_fact")} == before
    assert error_row["numeric_match"] is None  # failed requests stay unscored


def test_eval_md_answers_sections_match_the_stored_runs():
    """The committed report is re-derivable from the committed rows (no API call)."""
    eval_dir = REPO_ROOT / "data" / "eval"
    run_path, rerun_path = eval_dir / "answer_run.json", eval_dir / "answer_rerun.json"
    if not run_path.exists():
        return
    by_id = {i.id: i for i in load_golden(eval_dir / "golden.jsonl")}

    def stored(name):
        rows = list(load_done(eval_dir / name).values())
        rescore_numeric(rows, by_id)
        return rows

    eval_md = (REPO_ROOT / "docs" / "EVAL.md").read_text(encoding="utf-8")
    info, rows = json.loads(run_path.read_text("utf-8")), stored("answer_results.jsonl")
    assert render_answers_markdown(info, rows) in eval_md
    if rerun_path.exists():
        rerun_info = json.loads(rerun_path.read_text("utf-8"))
        current = render_current_markdown(info, rerun_info, rows, stored("answer_rerun.jsonl"))
        assert current in eval_md


def test_score_keeps_the_top_similarity():
    assert score(item(), {**GOOD, "top_score": 0.53712})["top_score"] == 0.537
    assert "top_score" not in score(item(), GOOD)


def test_rerun_replaces_rows_and_reports_both_outcomes():
    refused = {"status": "not_found", "answer": "Không tìm thấy", "citations": [], "sources": []}
    base = [
        score(item(qid="a"), refused),  # stopped at grade: model None
        score(item(qid="b"), GOOD),
        score(item(type_="unanswerable", expected=(), qid="u"), refused),
    ]
    rerun = [
        score(item(qid="a"), {**GOOD, "model": "m", "top_score": 0.537}),
        score(item(type_="unanswerable", expected=(), qid="u"),
              {**refused, "model": "m", "top_score": 0.501}),
    ]  # fmt: skip
    merged = merge_rerun(base, rerun)
    assert [r["id"] for r in merged] == ["a", "b", "u"]
    assert merged[0]["status"] == "answered" and merged[1] is base[1]

    base_info = {"date": "2026-11-24", "git": "abc"}
    rerun_info = {"date": "2026-11-25", "git": "def", "model": "m", "grade_min_score": 0.5,
                  "ids": ["a", "u"], "reason": "Threshold lowered.", "requests": 3,
                  "failures": 0, "stopped": None}  # fmt: skip
    md = render_current_markdown(base_info, rerun_info, base, rerun)
    assert "| `a` | numeric | 0.537 | refused at `grade`, no LLM call (wrong) |" in md
    assert "answered, cites BLLD2019/article/25 (right) |" in md
    assert "| `u` | unanswerable | 0.501 | refused at `grade`, no LLM call (right) | " in md
    assert "refused by the model (right) |" in md
    assert "eval-answers --ids a,u" in md and "failed: 0 (0.0%)" in md
    assert "| Answered (not refused) | 1.000 | 1.000 |" in md  # both answerable now answered
