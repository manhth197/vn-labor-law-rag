from __future__ import annotations

import json
import shutil
import subprocess
import sys

from tests.conftest import INTERIM_DIR, REPO_ROOT, word_count
from vnlaw_rag.config import Settings
from vnlaw_rag.ingest.pipeline import run_parse


def test_pipeline_runs_from_committed_interim_and_is_deterministic(tmp_path):
    shutil.copytree(INTERIM_DIR, tmp_path / "interim")
    settings = Settings(data_dir=tmp_path)  # no raw PDFs here: uses the interim lines

    info = run_parse(settings, count_tokens=word_count)
    first = {p.name: p.read_bytes() for p in (tmp_path / "processed").iterdir()}
    run_parse(settings, count_tokens=word_count)
    second = {p.name: p.read_bytes() for p in (tmp_path / "processed").iterdir()}

    assert set(first) == {"articles.jsonl", "appendices.jsonl", "chunks.jsonl", "build_info.json"}
    assert first == second
    assert info["documents"]["BLLD2019"]["articles"] == 220
    assert info["documents"]["ND145"]["articles"] == 115
    rows = (tmp_path / "processed" / "articles.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 335
    assert json.loads(rows[0])["article_title"] == "Phạm vi điều chỉnh"


def test_committed_chunks_match_the_architecture_rules():
    path = REPO_ROOT / "data" / "processed" / "chunks.jsonl"
    chunks = [json.loads(r) for r in path.read_text(encoding="utf-8").splitlines()]
    info = json.loads((REPO_ROOT / "data" / "processed" / "build_info.json").read_text("utf-8"))
    assert info["parameters"] == {"tokenizer": "BAAI/bge-m3", "chunk_max_tokens": 1200}
    assert all(c["n_tokens"] <= 1200 for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    for c in chunks:
        assert c["source_url"].startswith("https://vanban.chinhphu.vn/")
        assert c["doc_short"] in {"BLLD2019", "ND145"}


def test_serving_code_does_not_import_pymupdf():
    """PyMuPDF (AGPL) is an offline ingest tool only; importing the pipeline, parser and
    chunker must not load it."""
    code = (
        "import sys, vnlaw_rag.ingest.pipeline, vnlaw_rag.ingest.chunk, vnlaw_rag.cli; "
        "bad = {'pymupdf', 'fitz'} & set(sys.modules); assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
