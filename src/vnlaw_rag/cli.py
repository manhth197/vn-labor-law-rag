"""Command line entry point: `python -m vnlaw_rag.cli <command>`."""

from __future__ import annotations

import json
import logging

import typer

from vnlaw_rag.config import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )


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


if __name__ == "__main__":
    app()
