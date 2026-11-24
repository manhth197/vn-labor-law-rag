"""Command line entry point: `python -m vnlaw_rag.cli <command>`."""

from __future__ import annotations

import json
import logging
import sys

import typer

from vnlaw_rag.config import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    # Windows consoles and pipes default to a legacy code page; output is Vietnamese.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Request logs from the Hugging Face client are noise at INFO level.
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def fetch(
    force: bool = typer.Option(False, help="Download again even if the file is present."),
) -> None:
    """Download the source PDFs (Công báo) into data/raw and write manifest.json."""
    from vnlaw_rag.ingest.fetch import fetch_documents

    settings = get_settings()
    entries = fetch_documents(
        settings.raw_dir,
        settings.manifest_path,
        user_agent=settings.http_user_agent,
        delay_seconds=settings.http_delay_seconds,
        force=force,
    )
    for e in entries:
        status = "ok" if e.matches_pinned_sha256 else "CHANGED UPSTREAM"
        typer.echo(f"{e.path}: {e.bytes:,} bytes, sha256 {e.sha256[:12]}... {status}")


@app.command()
def parse() -> None:
    """Extract text, parse Chương/Mục/Điều/Khoản/Điểm and write articles and chunks."""
    from vnlaw_rag.ingest.pipeline import run_parse

    info = run_parse(get_settings())
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "anomalies"}
        for k, v in info["documents"].items()
    }
    typer.echo(
        json.dumps({"documents": summary, "chunks": info["chunks"]}, ensure_ascii=False, indent=2)
    )
    n_anomalies = sum(len(v["anomalies"]) for v in info["documents"].values())
    if n_anomalies:
        typer.echo(f"{n_anomalies} parser anomalies: see data/processed/build_info.json")


@app.command()
def index() -> None:
    """Embed data/processed/chunks.jsonl with bge-m3 and load it into pgvector."""
    from vnlaw_rag.index.build import build_index

    info = build_index(get_settings())
    typer.echo(json.dumps(info, ensure_ascii=False, indent=2))


@app.command()
def search(
    question: str,
    mode: str = typer.Option(None, help="vector | hybrid | keyword (default: RETRIEVAL_MODE)"),
    top_k: int = typer.Option(8, help="Number of chunks to show."),
) -> None:
    """Run retrieval for one question and print the ranked chunks."""
    from vnlaw_rag.retrieval.factory import open_retriever

    overrides = {"mode": mode} if mode else {}
    with open_retriever(get_settings(), **overrides) as retriever:
        for r in retriever.search(question, top_k=top_k):
            title = r.row.get("article_title") or r.row.get("appendix_no")
            typer.echo(
                f"{r.score:.4f}  {r.chunk_id:<22} v#{r.vector_rank or '-':<3} "
                f"k#{r.keyword_rank or '-':<3} {title}"
            )


@app.command()
def ask(
    question: str,
    show_sources: bool = typer.Option(False, help="Print the retrieved chunk ids and scores."),
) -> None:
    """Answer one question through the LangGraph flow (calls the Gemini API)."""
    from vnlaw_rag.graph.build import open_qa

    with open_qa(get_settings()) as qa:
        out = qa.invoke({"question": question})
    typer.echo(out["answer"])
    if out.get("citations"):
        typer.echo("\nTrích dẫn: " + ", ".join(out["citations"]))
    for warning in out.get("warnings", []):
        typer.echo(f"Lưu ý: {warning}")
    typer.echo(f"\n{out['notice']} Không phải tư vấn pháp lý.")
    if show_sources:
        for s in out.get("sources", []):
            score = s.get("vector_score")
            typer.echo(f"  {s['chunk_id']:<22} {score if score is None else round(score, 3)}")
    timings = ", ".join(f"{k} {v} ms" for k, v in out.get("timings_ms", {}).items())
    typer.echo(f"[{out['status']}; model {out.get('model')}; route {out['route']}; {timings}]")
    if out.get("dropped_citations"):
        typer.echo(f"[dropped citations: {', '.join(out['dropped_citations'])}]")


@app.command("eval")
def eval_retrieval(
    golden: str = typer.Option("data/eval/golden.jsonl", help="Golden set JSONL."),
) -> None:
    """Retrieval eval (no LLM): Hit@k, MRR@10 per configuration -> docs/EVAL.md."""

    from vnlaw_rag.config import REPO_ROOT
    from vnlaw_rag.eval.report import run_retrieval_eval, write_report

    settings = get_settings()
    run = run_retrieval_eval(settings, (REPO_ROOT / golden).resolve())
    write_report(
        run, REPO_ROOT / "docs" / "EVAL.md", REPO_ROOT / "data" / "eval" / "retrieval_results.json"
    )
    for label, summary in run["summary"].items():
        a = summary["all"]
        marker = "*" if label == run["default"] else " "
        typer.echo(
            f"{marker}{label:<25} hit@1 {a['hit@1']:.3f}  hit@5 {a['hit@5']:.3f}  "
            f"mrr@10 {a['mrr@10']:.3f}  recall@10 {a['recall@10']:.3f}"
        )
    typer.echo("wrote docs/EVAL.md and data/eval/retrieval_results.json")


@app.command("review-export")
def review_export(
    out: str = typer.Option("data/eval/review-golden.xlsx", help="Workbook to write."),
    n: int = typer.Option(25, help="Number of questions (stratified by type)."),
    seed: int = typer.Option(42, help="Sampling seed."),
    force: bool = typer.Option(False, help="Overwrite an existing workbook."),
) -> None:
    """Write an Excel workbook for the owner to review a sample of the golden set."""
    from vnlaw_rag.config import REPO_ROOT

    if (REPO_ROOT / out).exists() and not force:
        # The owner may be filling it in: never overwrite a review by accident.
        typer.echo(f"{out} exists (it may hold the owner's review); use --force to overwrite")
        raise typer.Exit(1)
    from vnlaw_rag.eval.review import export_review
    from vnlaw_rag.ingest.models import Appendix, Article

    processed = get_settings().processed_dir
    articles = _read_models(processed / "articles.jsonl", Article)
    appendices = _read_models(processed / "appendices.jsonl", Appendix)
    titles = {(a.doc_short, "article", str(a.article_no)): a.article_title for a in articles}
    titles |= {(p.doc_short, "appendix", p.appendix_no): p.title for p in appendices}
    # Excerpts keep one clause per line so they are easy to read in Excel.
    display = {
        (a.doc_short, "article", str(a.article_no)): f"{a.heading}\n{a.body_text()}"
        for a in articles
    }
    display |= {
        (p.doc_short, "appendix", p.appendix_no): f"Phụ lục {p.appendix_no}. {p.title}\n{p.text}"
        for p in appendices
    }
    sample = export_review(
        REPO_ROOT / "data" / "eval" / "golden.jsonl",
        display,
        titles,
        REPO_ROOT / out,
        n=n,
        seed=seed,
    )
    typer.echo(f"wrote {out} with {len(sample)} questions")


@app.command("review-import")
def review_import(
    workbook: str = typer.Argument(..., help="Filled workbook (sheet 'Duyệt')."),
) -> None:
    """Apply the owner's review (Đúng / Cần sửa / Bỏ) to data/eval/golden.jsonl."""
    from vnlaw_rag.config import REPO_ROOT
    from vnlaw_rag.eval.review import import_review

    eval_dir = REPO_ROOT / "data" / "eval"
    summary = import_review(
        REPO_ROOT / workbook, eval_dir / "golden.jsonl", eval_dir / "golden_removed.jsonl"
    )
    typer.echo(json.dumps(dict(summary.counts), ensure_ascii=False))
    if summary.edited:
        typer.echo(
            f"edited: {', '.join(summary.edited)}; run `pytest tests/test_eval.py` "
            "to re-check evidence quotes against the corpus"
        )


def _read_models(path, model):
    with path.open(encoding="utf-8") as fh:
        return [model.model_validate_json(line) for line in fh if line.strip()]


if __name__ == "__main__":
    app()
