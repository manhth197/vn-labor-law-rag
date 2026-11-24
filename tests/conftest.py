from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

from vnlaw_rag.ingest.models import Line, ParsedDocument
from vnlaw_rag.ingest.parse import parse_document
from vnlaw_rag.ingest.pipeline import interim_path, read_lines
from vnlaw_rag.ingest.sources import get_document

REPO_ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = REPO_ROOT / "data" / "interim"


@lru_cache
def parsed(doc_short: str) -> ParsedDocument:
    """Parse the committed interim lines of a real document (no PDF or network needed)."""
    doc = get_document(doc_short)
    return parse_document(read_lines(interim_path(INTERIM_DIR, doc)), doc)


def word_count(text: str) -> int:
    """Stand-in for the bge-m3 tokenizer in tests: one token per whitespace word."""
    return len(text.split())


@pytest.fixture
def blld() -> ParsedDocument:
    return parsed("BLLD2019")


@pytest.fixture
def nd145() -> ParsedDocument:
    return parsed("ND145")


def body(text: str, *, page: int = 1, part: int = 0) -> Line:
    """A normal body line starting at the left margin (a wrapped continuation)."""
    return Line(page=page, bold=False, x0=64, text=text, part=part)


def para(text: str, *, page: int = 1, part: int = 0) -> Line:
    """An indented line: the first line of a paragraph, clause or point."""
    return Line(page=page, bold=False, x0=87, text=text, part=part)


def bold(text: str, *, page: int = 1, part: int = 0, x0: int = 87) -> Line:
    return Line(page=page, bold=True, x0=x0, text=text, part=part)


@pytest.fixture(scope="session")
def rows():
    """Real chunks with real amendment annotations, as the database returns them."""
    from tests.fakes import build_rows

    return build_rows()
