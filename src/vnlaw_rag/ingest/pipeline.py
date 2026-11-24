"""Ingestion pipeline: PDF -> interim lines -> articles/appendices -> chunks.

Outputs are deterministic (no timestamps), so re-running on the same input gives the same
files and a clean `git diff`:

- data/interim/<DOC>.lines.jsonl     text lines with bold flag and x position (committed)
- data/processed/articles.jsonl      one record per Điều, clauses and points nested
- data/processed/appendices.jsonl    Phụ lục
- data/processed/chunks.jsonl        retrieval units with metadata
- data/processed/build_info.json     counts, parameters, input hashes, parser anomalies
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from vnlaw_rag.config import Settings
from vnlaw_rag.ingest.amendments import annotate_chunks, corpus_notice, load_registry
from vnlaw_rag.ingest.chunk import TokenCounter, build_chunks, load_token_counter
from vnlaw_rag.ingest.models import Line, ParsedDocument
from vnlaw_rag.ingest.parse import parse_document
from vnlaw_rag.ingest.sources import DOCUMENTS, SourceDocument

log = logging.getLogger(__name__)


def interim_path(interim_dir: Path, doc: SourceDocument) -> Path:
    return interim_dir / f"{doc.doc_short}.lines.jsonl"


def write_jsonl(path: Path, records: Iterable[BaseModel], exclude_defaults: bool = False) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(record.model_dump_json(exclude_defaults=exclude_defaults) + "\n")
            n += 1
    return n


def read_lines(path: Path) -> list[Line]:
    with path.open(encoding="utf-8") as fh:
        return [Line.model_validate_json(row) for row in fh if row.strip()]


def extract_or_load_lines(doc: SourceDocument, raw_dir: Path, interim_dir: Path) -> list[Line]:
    """Extract lines from the PDFs when they are present, else reuse the committed interim
    file. Extraction needs the optional `ingest` extra (PyMuPDF)."""
    pdfs = [raw_dir / f.filename for f in doc.files]
    target = interim_path(interim_dir, doc)
    if all(p.exists() for p in pdfs):
        try:
            from vnlaw_rag.ingest.pdf_text import extract_lines

            lines = [line for part, pdf in enumerate(pdfs) for line in extract_lines(pdf, part)]
        except RuntimeError as exc:
            if not target.exists():
                raise
            log.warning("%s; using %s", exc, target)
        else:
            write_jsonl(target, lines, exclude_defaults=True)  # keep the committed file small
            log.info("%s: extracted %d lines from %d PDF(s)", doc.doc_short, len(lines), len(pdfs))
            return lines
    elif not target.exists():
        raise FileNotFoundError(
            f"{doc.doc_short}: neither the PDFs nor {target} exist; run `fetch` first"
        )
    else:
        log.info("%s: PDFs not downloaded, using committed %s", doc.doc_short, target)
    return read_lines(target)


def run_parse(
    settings: Settings,
    documents: Iterable[SourceDocument] = DOCUMENTS,
    count_tokens: TokenCounter | None = None,
) -> dict:
    """Run the whole pipeline and return the build info that is also written to disk."""
    documents = list(documents)
    parsed: list[ParsedDocument] = []
    input_hashes: dict[str, str] = {}
    for doc in documents:
        lines = extract_or_load_lines(doc, settings.raw_dir, settings.interim_dir)
        parsed.append(parse_document(lines, doc))
        input_hashes[doc.doc_short] = _sha256(interim_path(settings.interim_dir, doc))

    articles = [a for p in parsed for a in p.articles]
    appendices = [a for p in parsed for a in p.appendices]
    count_tokens = count_tokens or load_token_counter(settings.tokenizer_name)
    chunks = build_chunks(articles, appendices, count_tokens, settings.chunk_max_tokens)
    registry_path = settings.data_dir / "amendments.json"
    registry = load_registry(registry_path)
    annotate_chunks(chunks, registry)

    out = settings.processed_dir
    write_jsonl(out / "articles.jsonl", articles)
    write_jsonl(out / "appendices.jsonl", appendices)
    write_jsonl(out / "chunks.jsonl", chunks)

    info = {
        "parameters": {
            "tokenizer": settings.tokenizer_name,
            "chunk_max_tokens": settings.chunk_max_tokens,
        },
        "inputs_sha256": input_hashes | {"amendments.json": _sha256(registry_path)},
        "corpus_notice": corpus_notice(registry, [d.text_version for d in documents]),
        "documents": {
            p.doc_short: {
                "articles": len(p.articles),
                "chapters": len({a.chapter_seq for a in p.articles}),
                "clauses": sum(len(a.clauses) for a in p.articles),
                "points": sum(
                    len(a.intro_points) + sum(len(c.points) for c in a.clauses) for a in p.articles
                ),
                "appendices": [f"{a.appendix_no} ({a.kind})" for a in p.appendices],
                "anomalies": [a.model_dump() for a in p.anomalies],
                "footnotes": p.footnotes,
            }
            for p in parsed
        },
        "chunks": {
            "total": len(chunks),
            "split_articles": sorted({c.chunk_id.split("-K")[0] for c in chunks if c.clause_nos}),
            "max_tokens_observed": max(c.n_tokens for c in chunks),
            "with_amended_by": sorted(c.chunk_id for c in chunks if c.amended_by),
            "with_upcoming_change": len([c for c in chunks if c.upcoming_change]),
        },
    }
    (out / "build_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return info


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
