from __future__ import annotations

import dataclasses
import hashlib
import json

import httpx

from vnlaw_rag.ingest.fetch import fetch_documents
from vnlaw_rag.ingest.sources import BLLD2019, SourceFile

CONTENT = b"%PDF-1.4 fake gazette"


def make_doc(sha256: str):
    src = SourceFile(url="https://example.invalid/a.pdf", filename="a.pdf", sha256=sha256)
    return dataclasses.replace(BLLD2019, files=(src,))


def client_counting(calls: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        assert "vnlaw-rag" in request.headers["user-agent"]
        return httpx.Response(200, content=CONTENT)

    return httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": "vnlaw-rag"})


def test_fetch_downloads_and_writes_manifest(tmp_path):
    calls: list[str] = []
    doc = make_doc(hashlib.sha256(CONTENT).hexdigest())
    manifest = tmp_path / "manifest.json"
    entries = fetch_documents(
        tmp_path, manifest, [doc], user_agent="x", delay_seconds=0, client=client_counting(calls)
    )
    assert calls == ["https://example.invalid/a.pdf"]
    assert (tmp_path / "a.pdf").read_bytes() == CONTENT
    (entry,) = entries
    assert entry.matches_pinned_sha256
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["files"][0]["sha256"] == hashlib.sha256(CONTENT).hexdigest()
    assert data["files"][0]["url"] == "https://example.invalid/a.pdf"
    assert data["files"][0]["retrieved_at"]


def test_fetch_skips_files_already_present(tmp_path):
    calls: list[str] = []
    doc = make_doc(hashlib.sha256(CONTENT).hexdigest())
    (tmp_path / "a.pdf").write_bytes(CONTENT)
    fetch_documents(
        tmp_path,
        tmp_path / "m.json",
        [doc],
        user_agent="x",
        delay_seconds=0,
        client=client_counting(calls),
    )
    assert calls == []


def test_fetch_flags_upstream_change(tmp_path):
    doc = make_doc("0" * 64)
    (entry,) = fetch_documents(
        tmp_path, tmp_path / "m.json", [doc], user_agent="x", delay_seconds=0,
        client=client_counting([]),
    )  # fmt: skip
    assert not entry.matches_pinned_sha256
