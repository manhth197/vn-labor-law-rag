"""Run the retrieval evaluation for several configurations and write docs/EVAL.md."""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from vnlaw_rag.config import REPO_ROOT, Settings
from vnlaw_rag.eval.golden import GoldenItem, load_golden
from vnlaw_rag.eval.retrieval import (
    CachedEmbeddings,
    QuestionResult,
    evaluate,
    score_separation,
    summarize,
)
from vnlaw_rag.index.embed import load_embeddings
from vnlaw_rag.index.store import PgStore
from vnlaw_rag.retrieval.hybrid import HybridRetriever

# (label, retrieval mode, keyword terms, keyword scoring, drop stop words)
CONFIGS = [
    ("vector", "vector", "bigram", "idf", True),
    ("keyword-bigram-tsrank", "keyword", "bigram", "ts_rank_cd", False),
    ("keyword-bigram-idf", "keyword", "bigram", "idf", False),
    ("keyword-bigram-idf-stop", "keyword", "bigram", "idf", True),
    ("keyword-unigram-idf-stop", "keyword", "unigram", "idf", True),
    ("hybrid-bigram-tsrank", "hybrid", "bigram", "ts_rank_cd", False),
    ("hybrid-bigram-idf", "hybrid", "bigram", "idf", False),
    ("hybrid-bigram-idf-stop", "hybrid", "bigram", "idf", True),
    ("hybrid-unigram-idf-stop", "hybrid", "unigram", "idf", True),
]


def default_label(settings: Settings) -> str:
    """The CONFIGS label that matches the app's configured retrieval."""
    if settings.retrieval_mode == "vector":
        return "vector"
    scoring = "tsrank" if settings.keyword_scoring == "ts_rank_cd" else "idf"
    stop = "-stop" if settings.keyword_stopwords else ""
    return f"{settings.retrieval_mode}-{settings.keyword_mode}-{scoring}{stop}"


START, END = "<!-- retrieval:start -->", "<!-- retrieval:end -->"


def git_revision() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    sha = git("rev-parse", "--short", "HEAD")
    return sha + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def run_retrieval_eval(settings: Settings, golden_path: Path) -> dict:
    items = load_golden(golden_path)
    embeddings = CachedEmbeddings(load_embeddings(settings))
    store = PgStore.open(settings.database_url)
    try:
        index_info = store.index_info()
        runs: dict[str, list[QuestionResult]] = {}
        for label, mode, keyword_mode, scoring, stopwords in CONFIGS:
            retriever = HybridRetriever(
                store=store,
                embeddings=embeddings,
                mode=mode,
                keyword_mode=keyword_mode,
                keyword_scoring=scoring,
                keyword_stopwords=stopwords,
                vector_k=settings.retrieval_vector_k,
                keyword_k=settings.retrieval_keyword_k,
                rrf_k=settings.rrf_k,
            )
            runs[label] = evaluate(retriever, items)
    finally:
        store.close()
    reviewed = {i.id: i.reviewed for i in items}
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "default": default_label(settings),
        "git": git_revision(),
        "settings": {
            "embedding_model": settings.embedding_model,
            "vector_k": settings.retrieval_vector_k,
            "keyword_k": settings.retrieval_keyword_k,
            "rrf_k": settings.rrf_k,
            "chunk_max_tokens": settings.chunk_max_tokens,
        },
        "index": {k: index_info.get(k) for k in ("chunks", "chunks_sha256", "indexed_at")},
        "golden": {
            "path": golden_path.relative_to(REPO_ROOT).as_posix(),
            "n": len(items),
            "types": dict(Counter(i.type for i in items)),
            "reviewed": sum(i.reviewed for i in items),
        },
        "summary": {label: summarize(rs) for label, rs in runs.items()},
        "summary_reviewed": {
            label: summarize([r for r in rs if reviewed[r.id]]) for label, rs in runs.items()
        },
        "separation": score_separation(runs["vector"]),
        "results": {label: [asdict(r) for r in rs] for label, rs in runs.items()},
        "items": {i.id: i for i in items},
    }


def render_markdown(run: dict) -> str:
    s, g = run["settings"], run["golden"]
    types = ["lookup", "paraphrase", "numeric", "multi"]
    lines = [
        START,
        "## Retrieval",
        "",
        f"Run: {run['date']} · git `{run['git']}` · command `python -m vnlaw_rag.cli eval`",
        "",
        f"- Embeddings: `{s['embedding_model']}` (normalized, cosine, HNSW). "
        f"Chunks: {run['index']['chunks']} (max {s['chunk_max_tokens']} tokens), "
        f"chunks.jsonl sha256 `{str(run['index']['chunks_sha256'])[:12]}`.",
        f"- Candidates: vector top {s['vector_k']}, keyword top {s['keyword_k']}; "
        f"RRF k = {s['rrf_k']}. Metrics at depth 10, article-level matching.",
        f"- Golden set: `{g['path']}`, {g['n']} questions "
        f"({', '.join(f'{k} {v}' for k, v in g['types'].items())}). "
        f"**Reviewed by the owner: {g['reviewed']}/{g['n']}.** Unanswerable questions are "
        "excluded from the ranking metrics.",
        "",
        "> **Caveat (golden set do agent soạn, có thể thiên lệch).** The questions were drafted "
        "by the coding agent from the article texts, not by users. Lookup items reuse the law's "
        "wording, and there are few keyword-heavy questions (exact terms, document numbers). "
        f"The owner reviewed a stratified sample of 25 and accepted {g['reviewed']}; the other "
        "questions are still unreviewed.",
        "",
        "### Reviewed subset (owner-reviewed answerable questions)",
        "",
        f"Small sample: n = {run['summary_reviewed']['vector']['all']['n']}, so one question "
        f"moves Hit@k by {100 / run['summary_reviewed']['vector']['all']['n']:.1f} points. "
        "No unanswerable question is in the reviewed subset.",
        "",
        "| Configuration | n | Hit@1 | Hit@5 | MRR@10 | Recall@10 |",
        "|---|---|---|---|---|---|",
        *_overall_rows(run, "summary_reviewed"),
        "",
        "Configuration names: `keyword-bigram-tsrank` = OR of syllable pairs ranked by "
        "`ts_rank_cd`; `-idf` = BM25-style IDF weighting computed in SQL; `-stop` = terms with "
        "Vietnamese function words dropped; `hybrid-*` = that keyword leg fused with the "
        "vector leg by RRF.",
        "",
        "### All answerable questions (reviewed + unreviewed)",
        "",
        "| Configuration | n | Hit@1 | Hit@5 | MRR@10 | Recall@10 |",
        "|---|---|---|---|---|---|",
        *_overall_rows(run, "summary"),
    ]
    lines += [
        "",
        "### By question type: Hit@5 / MRR@10",
        "",
        "| Configuration | "
        + " | ".join(f"{t} (n={run['summary']['vector'][t]['n']})" for t in types)
        + " |",
        "|---|" + "---|" * len(types),
    ]
    for label, summary in run["summary"].items():
        cells = [f"{summary[t]['hit@5']:.2f} / {summary[t]['mrr@10']:.2f}" for t in types]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    sep = run["separation"]
    lines += [
        "",
        "### Top-1 cosine similarity: answerable vs unanswerable",
        "",
        "| Questions | n | min | median | max |",
        "|---|---|---|---|---|",
    ]
    for label, v in sep.items():
        lines.append(
            f"| {label} | {v['n']} | {v['min']:.3f} | {v['median']:.3f} | {v['max']:.3f} |"
        )
    default = run["default"]
    lines += ["", f"### Misses of the default configuration ({default}, not in top 5)", ""]
    misses = [
        r for r in run["results"][run["default"]] if r["type"] != "unanswerable" and not _hit5(r)
    ]
    if not misses:
        lines.append("None.")
    for r in misses:
        item: GoldenItem = run["items"][r["id"]]
        expected = ", ".join("/".join(ref.key()) for ref in item.expected)
        rank = r["first_relevant_rank"] or "not in top 10"
        lines.append(
            f"- `{r['id']}` ({r['type']}) {item.question} Expected {expected}; "
            f"first relevant rank: {rank}; top 3: {', '.join(r['ranked_units'][:3])}"
        )
    lines += [END]
    return "\n".join(lines) + "\n"


def _overall_rows(run: dict, key: str) -> list[str]:
    rows = []
    for label, summary in run[key].items():
        a = summary["all"]
        name = f"**{label}** (default)" if label == run["default"] else label
        rows.append(
            f"| {name} | {a['n']} | {a['hit@1']:.3f} | {a['hit@5']:.3f} | {a['mrr@10']:.3f} "
            f"| {a['recall@10']:.3f} |"
        )
    return rows


def _hit5(result: dict) -> bool:
    rank = result["first_relevant_rank"]
    return rank is not None and rank <= 5


def write_report(run: dict, eval_md: Path, results_json: Path) -> None:
    section = render_markdown(run)
    if eval_md.exists():
        text = eval_md.read_text(encoding="utf-8")
        if START in text and END in text:
            before, rest = text.split(START, 1)
            after = rest.split(END, 1)[1].lstrip("\n")
            text = before + section + ("\n" + after if after else "")
        else:
            text = text.rstrip() + "\n\n" + section
    else:
        text = (
            "# Evaluation\n\nNo invented numbers: every table below is produced by a "
            "command and records its date, git revision and configuration.\n\n" + section
        )
    eval_md.write_text(text, encoding="utf-8", newline="\n")
    payload = {k: v for k, v in run.items() if k != "items"}
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
