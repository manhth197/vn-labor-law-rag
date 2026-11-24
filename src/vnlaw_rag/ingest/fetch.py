"""Download the source PDFs politely and record them in `data/raw/manifest.json`."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from vnlaw_rag.ingest.sources import DOCUMENTS, SourceDocument

log = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    doc_short: str
    doc_number: str
    url: str
    source_url: str
    path: str  # relative to the raw directory
    retrieved_at: str
    sha256: str
    bytes: int
    matches_pinned_sha256: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {e["path"]: ManifestEntry(**e) for e in data["files"]}


def write_manifest(path: Path, entries: Iterable[ManifestEntry]) -> None:
    payload = {"files": [asdict(e) for e in sorted(entries, key=lambda e: e.path)]}
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def fetch_documents(
    raw_dir: Path,
    manifest_path: Path,
    documents: Iterable[SourceDocument] = DOCUMENTS,
    *,
    user_agent: str,
    delay_seconds: float = 1.0,
    force: bool = False,
    client: httpx.Client | None = None,
) -> list[ManifestEntry]:
    """Download every file of `documents` into `raw_dir`, skipping files already present
    with the pinned hash unless `force` is set. Returns the manifest entries written."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    documents = list(documents)
    # Forget files that are no longer sources (e.g. a text replaced by its consolidated one).
    known = {f.filename for doc in (*DOCUMENTS, *documents) for f in doc.files}
    manifest = {k: v for k, v in load_manifest(manifest_path).items() if k in known}
    own_client = client is None
    client = client or httpx.Client(
        headers={"User-Agent": user_agent}, timeout=120.0, follow_redirects=True
    )
    downloaded_any = False
    try:
        for doc in documents:
            for src in doc.files:
                target = raw_dir / src.filename
                if target.exists() and not force and sha256_file(target) == src.sha256:
                    log.info("skip %s (already present, hash matches)", src.filename)
                    if src.filename not in manifest:
                        # File was placed by hand; its mtime is the best "retrieved" time.
                        mtime = datetime.fromtimestamp(target.stat().st_mtime, timezone.utc)
                        manifest[src.filename] = _entry(
                            doc, src.url, target, mtime.replace(microsecond=0).isoformat()
                        )
                    continue
                if downloaded_any:
                    time.sleep(delay_seconds)  # be gentle with the government portal
                log.info("download %s", src.url)
                response = client.get(src.url)
                response.raise_for_status()
                target.write_bytes(response.content)
                downloaded_any = True
                entry = _entry(doc, src.url, target, _now())
                if not entry.matches_pinned_sha256:
                    log.warning(
                        "%s: sha256 %s differs from pinned %s; the upstream file changed, "
                        "re-check the parser output before using it",
                        src.filename,
                        entry.sha256,
                        src.sha256,
                    )
                manifest[src.filename] = entry
    finally:
        if own_client:
            client.close()
    write_manifest(manifest_path, manifest.values())
    return list(manifest.values())


def _entry(doc: SourceDocument, url: str, target: Path, retrieved_at: str) -> ManifestEntry:
    digest = sha256_file(target)
    pinned = next(f.sha256 for f in doc.files if f.filename == target.name)
    return ManifestEntry(
        doc_short=doc.doc_short,
        doc_number=doc.doc_number,
        url=url,
        source_url=doc.source_url,
        path=target.name,
        retrieved_at=retrieved_at,
        sha256=digest,
        bytes=target.stat().st_size,
        matches_pinned_sha256=digest == pinned,
    )


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
