"""Answer evaluation: deterministic metrics first, an LLM judge as a secondary signal.

Deterministic (primary):
- citation precision / recall: cited articles vs expected articles (article level);
- refusal correctness: unanswerable questions must get "không tìm thấy", answerable ones
  must not (false refusals);
- numeric exact match: for `numeric` questions, the key figure of the reference answer (the
  first number with a unit, e.g. "60 ngày", "85%") must appear in the answer, after
  normalizing digits ("05" = "5", "2.000.000.000" = "2 tỷ") and simple number words.

Secondary: a reference-based LLM judge scores correctness/completeness and faithfulness to
the cited sources (1-5). It is the same model as the generator, so it may favor its own
answers; the report says so.

The run is sequential with a pause between API requests, appends one JSON line per question
(so it can resume after an interruption), stops when more than 20% of the first 20 API
requests fail, and stops on a persistent rate limit, reporting partial results with their
sample size.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vnlaw_rag.eval.golden import GoldenItem, UnitKey
from vnlaw_rag.llm.base import LLMError, RateLimited

log = logging.getLogger(__name__)

# -- numeric facts -------------------------------------------------------------------------
UNITS = ("%", "ngày", "tháng", "năm", "giờ", "tuổi", "phút", "đồng", "người", "lần")
_UNIT = "|".join(re.escape(u) for u in UNITS)
_NUMBER_FACT = re.compile(
    rf"(?<![\w.,])(\d{{1,3}}(?:\.\d{{3}})+|\d+(?:,\d+)?)\s*(tỷ\s*)?({_UNIT})(?![\w])", re.IGNORECASE
)
_WORDS = {
    "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5, "sáu": 6, "bảy": 7, "tám": 8, "chín": 9,
    "mười": 10,
}  # fmt: skip
_WORD_FACT = re.compile(
    rf"\b(mười\s+(?:một|hai|ba|bốn|lăm|năm|sáu|bảy|tám|chín)|{'|'.join(_WORDS)})\s+({_UNIT})\b",
    re.IGNORECASE,
)


def _number(text: str) -> str:
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", text):
        text = text.replace(".", "")
    text = text.replace(",", ".")
    if "." in text:
        return str(float(text)).rstrip("0").rstrip(".")
    return str(int(text))


def _word_number(words: str) -> int:
    parts = words.lower().split()
    if len(parts) == 2:  # "mười hai" = 12, "mười lăm" = 15
        unit = 5 if parts[1] == "lăm" else _WORDS[parts[1]]
        return 10 + unit
    return _WORDS[parts[0]]


def facts(text: str) -> list[tuple[str, str]]:
    """(number, unit) pairs in a text, normalized; "2 tỷ đồng" -> ("2000000000", "đồng")."""
    out: list[tuple[str, str]] = []
    for value, billion, unit in _NUMBER_FACT.findall(text):
        number = _number(value)
        if billion:
            number = str(int(float(number) * 1_000_000_000))
        out.append((number, unit.lower()))
    for words, unit in _WORD_FACT.findall(text):
        out.append((str(_word_number(words)), unit.lower()))
    return out


def key_fact(reference_answer: str) -> tuple[str, str] | None:
    """The key figure of a reference answer: its first number with a unit."""
    for match in _NUMBER_FACT.finditer(reference_answer):
        return facts(match.group(0))[0]
    return None


# -- citations -----------------------------------------------------------------------------
def cited_units(citations: Iterable[str], sources: Iterable[dict[str, Any]]) -> list[UnitKey]:
    by_id = {s["chunk_id"]: s for s in sources}
    units: list[UnitKey] = []
    for chunk_id in citations:
        s = by_id.get(chunk_id)
        if s is None:
            continue
        unit = (
            (s["doc_short"], "appendix", str(s["appendix_no"]))
            if s.get("unit") == "appendix"
            else (s["doc_short"], "article", str(s["article_no"]))
        )
        if unit not in units:
            units.append(unit)
    return units


# -- judge ---------------------------------------------------------------------------------
JUDGE_SYSTEM = """\
Bạn là giám khảo chấm câu trả lời về pháp luật lao động Việt Nam. Bạn nhận CÂU HỎI, ĐÁP ÁN \
THAM CHIẾU (đúng), CÂU TRẢ LỜI cần chấm và các NGUỒN mà câu trả lời trích dẫn.
Chấm hai điểm, mỗi điểm từ 1 đến 5:
- correctness: câu trả lời có đúng và đủ ý chính so với ĐÁP ÁN THAM CHIẾU không. 5 = đúng và \
đủ; 4 = đúng, thiếu chi tiết nhỏ; 3 = đúng một phần hoặc thiếu ý quan trọng; 2 = phần lớn sai; \
1 = sai hoặc không trả lời.
- faithfulness: mọi khẳng định trong câu trả lời có dựa trên NGUỒN không. 5 = hoàn toàn dựa \
trên nguồn; 3 = có ý không thấy trong nguồn; 1 = bịa hoặc mâu thuẫn với nguồn.
Chỉ chấm, không sửa câu trả lời. Lý do ngắn gọn, tiếng Việt."""


class JudgeScore(BaseModel):
    correctness: int = Field(ge=1, le=5)
    faithfulness: int = Field(ge=1, le=5)
    rationale: str


def judge_prompt(item: GoldenItem, answer: str, cited_sources: list[dict[str, Any]]) -> str:
    sources = "\n\n".join(f"[{s['chunk_id']}]\n{s['content']}" for s in cited_sources)
    return (
        f"CÂU HỎI: {item.question}\n\nĐÁP ÁN THAM CHIẾU: {item.reference_answer}\n\n"
        f"CÂU TRẢ LỜI CẦN CHẤM: {answer}\n\nNGUỒN:\n{sources or '(không có)'}"
    )


# -- scoring one question ------------------------------------------------------------------
def score(item: GoldenItem, out: dict[str, Any]) -> dict[str, Any]:
    """Deterministic metrics for one graph output."""
    refused = out.get("status") != "answered"
    row: dict[str, Any] = {
        "id": item.id,
        "type": item.type,
        "reviewed": item.reviewed,
        "status": out.get("status"),
        "answer": out.get("answer", ""),
        "citations": out.get("citations", []),
        "model": out.get("model"),
        "refusal_ok": refused if not item.answerable else not refused,
    }
    if item.answerable and not refused:
        cited = cited_units(out.get("citations", []), out.get("sources", []))
        expected = {r.key() for r in item.expected}
        hits = [u for u in cited if u in expected]
        row["cited_units"] = ["/".join(u) for u in cited]
        row["precision"] = len(hits) / len(cited) if cited else 0.0
        row["recall"] = len(set(hits)) / len(expected)
    if item.type == "numeric":
        fact = key_fact(item.reference_answer)
        row["key_fact"] = " ".join(fact) if fact else None
        row["numeric_match"] = bool(fact) and not refused and fact in facts(row["answer"])
    return row


# -- run -----------------------------------------------------------------------------------
@dataclass
class RequestLog:
    """Outcome of every API request, in order, across generator and judge."""

    outcomes: list[bool] = field(default_factory=list)

    def __call__(self, model: str, ok: bool) -> None:
        self.outcomes.append(ok)

    @property
    def failures(self) -> int:
        return self.outcomes.count(False)

    def first_window_failure_rate(self, window: int = 20) -> float | None:
        head = self.outcomes[:window]
        return head.count(False) / len(head) if len(head) >= window else None


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {r["id"]: r for r in rows}


def run_answer_eval(
    items: list[GoldenItem],
    graph: Any,
    judge: Any,
    results_path: Path,
    requests: RequestLog,
    pause_seconds: float = 4.0,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Answer and score every item not already in `results_path`. Returns run info."""
    done = load_done(results_path)
    stopped: str | None = None
    for item in items:
        if item.id in done:
            continue
        before = len(requests.outcomes)
        started = time.perf_counter()
        try:
            out = graph.invoke({"question": item.question})
        except RateLimited as exc:
            stopped = f"rate limited at {item.id}: {exc}"
            break
        except LLMError as exc:
            row = {"id": item.id, "type": item.type, "reviewed": item.reviewed,
                   "status": "error", "error": str(exc)[:200]}  # fmt: skip
            _append(results_path, row)
            if _too_many_failures(requests):
                stopped = "more than 20% of the first 20 requests failed"
                break
            continue
        row = score(item, out)
        row["seconds"] = round(time.perf_counter() - started, 2)
        if len(requests.outcomes) > before:
            sleep(pause_seconds)  # the generator called the API: pace the next request
        if item.answerable and row["status"] == "answered":
            cited = [s for s in out["sources"] if s["chunk_id"] in out["citations"]]
            try:
                verdict = judge.generate_structured(
                    system=JUDGE_SYSTEM,
                    prompt=judge_prompt(item, out["answer"], cited),
                    schema=JudgeScore,
                ).value
                row["judge"] = verdict.model_dump()
            except RateLimited as exc:
                _append(results_path, row)
                stopped = f"rate limited while judging {item.id}: {exc}"
                break
            except LLMError as exc:
                row["judge_error"] = str(exc)[:200]
            sleep(pause_seconds)
        _append(results_path, row)
        if _too_many_failures(requests):
            stopped = "more than 20% of the first 20 requests failed"
            break
    return {"stopped": stopped, "requests": len(requests.outcomes), "failures": requests.failures}


def _too_many_failures(requests: RequestLog) -> bool:
    rate = requests.first_window_failure_rate()
    return rate is not None and rate > 0.2


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# -- summary -------------------------------------------------------------------------------
def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 3) if values else None


def summarize_answers(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["type"] != "unanswerable" and r["status"] != "error"]
    answered = [r for r in answerable if r["status"] == "answered"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable" and r["status"] != "error"]
    numeric = [r for r in answerable if r["type"] == "numeric"]
    judged = [r for r in answered if "judge" in r]
    return {
        "n_rows": len(rows),
        "errors": sum(1 for r in rows if r["status"] == "error"),
        "answerable": len(answerable),
        "answered_rate": _mean([1.0 if r["status"] == "answered" else 0.0 for r in answerable]),
        "false_refusal_rate": _mean([0.0 if r["refusal_ok"] else 1.0 for r in answerable]),
        "citation_precision": _mean([r["precision"] for r in answered]),
        "citation_recall": _mean([r["recall"] for r in answered]),
        "cites_expected_article": _mean([1.0 if r["recall"] > 0 else 0.0 for r in answered]),
        "numeric_n": len(numeric),
        "numeric_exact_match": _mean([1.0 if r.get("numeric_match") else 0.0 for r in numeric]),
        "unanswerable": len(unanswerable),
        "correct_refusal_rate": _mean([1.0 if r["refusal_ok"] else 0.0 for r in unanswerable]),
        "judged": len(judged),
        "judge_correctness": _mean([r["judge"]["correctness"] for r in judged]),
        "judge_faithfulness": _mean([r["judge"]["faithfulness"] for r in judged]),
    }


# -- report --------------------------------------------------------------------------------
ANSWERS_START, ANSWERS_END = "<!-- answers:start -->", "<!-- answers:end -->"
METRIC_ROWS = [
    ("Answerable questions scored", "answerable", "{}"),
    ("Answered (not refused)", "answered_rate", "{:.3f}"),
    ("False refusals", "false_refusal_rate", "{:.3f}"),
    ("Citation precision (article level)", "citation_precision", "{:.3f}"),
    ("Citation recall (article level)", "citation_recall", "{:.3f}"),
    ("Cites at least one expected article", "cites_expected_article", "{:.3f}"),
    ("Numeric questions", "numeric_n", "{}"),
    ("Numeric exact match (key figure)", "numeric_exact_match", "{:.3f}"),
    ("Unanswerable questions", "unanswerable", "{}"),
    ("Unanswerable correctly refused", "correct_refusal_rate", "{:.3f}"),
    ("Judged answers", "judged", "{}"),
    ("Judge: correctness 1-5 (secondary)", "judge_correctness", "{:.2f}"),
    ("Judge: faithfulness 1-5 (secondary)", "judge_faithfulness", "{:.2f}"),
]


def render_answers_markdown(info: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    reviewed = summarize_answers([r for r in rows if r.get("reviewed")])
    everything = summarize_answers(rows)
    requests, failures = info["requests"], info["failures"]
    rate = f"{failures / requests:.1%}" if requests else "n/a"
    lines = [
        ANSWERS_START,
        "## Answers",
        "",
        f"Run: {info['date']} · git `{info['git']}` · command "
        "`python -m vnlaw_rag.cli eval-answers`",
        "",
        f"- Generator: `{info['model']}` (main model only, no fallback, 180 s budget); "
        f"judge: `{info['judge_model']}`. Sequential, {info['pause_s']:.0f} s between API "
        "requests.",
        f"- API requests: {requests} (generator + judge), failed: {failures} ({rate}). "
        f"Questions scored: {len(rows)} of {info['golden_n']}"
        + (f". **Stopped early: {info['stopped']}**" if info.get("stopped") else "")
        + ".",
        "",
        "> **Caveats.** The judge is the same model as the generator, so it may favor its own "
        "answers: the deterministic rows are the main numbers. The golden set was drafted by "
        "the coding agent; the owner reviewed 22 answerable questions. Numeric match checks "
        "only the first figure of the reference answer.",
        "",
        f"| Metric | Reviewed (n={reviewed['n_rows']}) | All (n={everything['n_rows']}) |",
        "|---|---|---|",
    ]
    for label, key, fmt in METRIC_ROWS:
        cells = [fmt.format(s[key]) if s[key] is not None else "-" for s in (reviewed, everything)]
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")
    misses = [
        r for r in rows
        if r["status"] == "error" or not r.get("refusal_ok", True)
        or (r["type"] == "numeric" and not r.get("numeric_match"))
    ]  # fmt: skip
    lines += ["", "### Errors, wrong refusals and numeric misses", ""]
    if not misses:
        lines.append("None.")
    for r in misses:
        why = []
        if r["status"] == "error":
            why.append(f"error: {r.get('error', '')[:80]}")
        elif not r.get("refusal_ok", True):
            why.append(
                "wrong refusal" if r["type"] != "unanswerable" else "answered, should refuse"
            )
        if r["type"] == "numeric" and not r.get("numeric_match") and r["status"] != "error":
            why.append(f"key figure '{r.get('key_fact')}' missing")
        lines.append(f"- `{r['id']}` ({r['type']}): {'; '.join(why)}")
    lines += ["", "Per-question answers and scores: `data/eval/answer_results.jsonl`.", ANSWERS_END]
    return "\n".join(lines) + "\n"
