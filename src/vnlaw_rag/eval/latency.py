"""Measure LLM latency and reliability per model on a few reviewed golden questions.

Each call is a single streaming attempt (no retry, no fallback) with the app's per-call
timeout, so failures are counted, not hidden. Models are interleaved per question to share
the same conditions, with a pause between calls to stay under free-tier per-minute limits.
Time to first answer token (TTFT) matters most for the UI; total time for the full answer.

The prompts are rebuilt without loading the embedding model (2-3 GB of RAM): the sources are
the chunks of the articles ranked first by the default (vector) configuration in the last
retrieval eval (data/eval/retrieval_results.json), in that order. Only the LLM is measured.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from vnlaw_rag.config import Settings
from vnlaw_rag.eval.golden import load_golden
from vnlaw_rag.eval.report import git_revision
from vnlaw_rag.graph.prompts import SYSTEM_PROMPT, GeneratedAnswer, build_prompt
from vnlaw_rag.index.build import load_chunks
from vnlaw_rag.ingest.amendments import annotate_chunks, load_registry
from vnlaw_rag.llm.gemini import GeminiClient, status_of

DEFAULT_QUESTIONS = ("q004", "q025", "q039", "q058", "q034")


@dataclass
class Sample:
    model: str
    question_id: str
    ok: bool
    ttft_s: float | None
    total_s: float
    error: str | None = None


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile (with 5 samples, p95 is the maximum)."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, round(q / 100 * len(ordered) + 0.5))
    return ordered[min(rank, len(ordered)) - 1]


def recorded_sources(settings: Settings, question_ids) -> dict[str, list[dict]]:
    """Sources per question from the recorded vector ranking (article level) and chunks.jsonl."""
    run = json.loads((settings.data_dir / "eval" / "retrieval_results.json").read_text("utf-8"))
    ranked = {r["id"]: r["ranked_units"] for r in run["results"]["vector"]}
    chunks = load_chunks(settings.processed_dir / "chunks.jsonl")
    annotate_chunks(chunks, load_registry(settings.data_dir / "amendments.json"))
    by_unit: dict[str, list[dict]] = {}
    for c in chunks:
        unit = (
            f"{c.doc_short}/appendix/{c.appendix_no}"
            if c.unit == "appendix"
            else f"{c.doc_short}/article/{c.article_no}"
        )
        by_unit.setdefault(unit, []).append({**c.model_dump(), "content": c.text})
    out = {}
    for qid in question_ids:
        rows = [row for unit in ranked[qid] for row in by_unit.get(unit, [])]
        out[qid] = rows[: settings.llm_max_sources]
    return out


def run_latency_bench(
    settings: Settings,
    models: list[str],
    question_ids: tuple[str, ...] = DEFAULT_QUESTIONS,
    pause_seconds: float = 4.0,
    on_sample: Callable[[Sample], None] | None = None,
) -> dict:
    items = {i.id: i for i in load_golden(settings.data_dir / "eval" / "golden.jsonl")}
    key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else ""
    samples: list[Sample] = []
    sources = recorded_sources(settings, question_ids)
    prompts = {qid: build_prompt(items[qid].question, sources[qid]) for qid in question_ids}
    clients = {
        m: GeminiClient(
            api_key=key,
            models=[m],
            thinking_level=settings.llm_thinking_level,
            max_output_tokens=settings.llm_max_output_tokens,
            call_timeout_seconds=settings.llm_timeout_seconds,
            deadline_seconds=settings.llm_timeout_seconds + 5,
            max_attempts=1,
        )  # fmt: skip
        for m in models
    }
    for qid in question_ids:
        for model in models:
            if samples:
                time.sleep(pause_seconds)
            first: list[float] = []
            started = time.perf_counter()

            def on_text(_: str, started=started, first=first) -> None:
                if not first:
                    first.append(time.perf_counter() - started)

            try:
                clients[model].generate_structured(
                    system=SYSTEM_PROMPT, prompt=prompts[qid], schema=GeneratedAnswer,
                    on_text=on_text,
                )  # fmt: skip
                error = None
            except Exception as exc:  # recorded, not raised: failures are the point
                cause = exc.__cause__ or exc
                error = f"{type(cause).__name__} (status {status_of(cause)})"
            total = time.perf_counter() - started
            samples.append(
                Sample(model, qid, error is None, round(first[0], 2) if first else None,
                       round(total, 2), error)
            )  # fmt: skip
            if on_sample:
                on_sample(samples[-1])

    summary = {}
    for model in models:
        mine = [s for s in samples if s.model == model]
        ok = [s for s in mine if s.ok]
        totals = [s.total_s for s in ok]
        ttfts = [s.ttft_s for s in ok if s.ttft_s is not None]
        summary[model] = {
            "n": len(mine),
            "errors": len(mine) - len(ok),
            "error_rate": round((len(mine) - len(ok)) / len(mine), 2) if mine else None,
            "ttft_p50_s": percentile(ttfts, 50),
            "total_p50_s": percentile(totals, 50),
            "total_p95_s": percentile(totals, 95),
            "mean_total_s": round(statistics.fmean(totals), 2) if totals else None,
        }
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git": git_revision(),
        "settings": {
            "call_timeout_s": settings.llm_timeout_seconds,
            "thinking_level": settings.llm_thinking_level,
            "max_sources": settings.llm_max_sources,
            "pause_s": pause_seconds,
        },
        "questions": list(question_ids),
        "summary": summary,
        "samples": [asdict(s) for s in samples],
    }


def write_latency(result: dict, path: Path) -> None:
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
