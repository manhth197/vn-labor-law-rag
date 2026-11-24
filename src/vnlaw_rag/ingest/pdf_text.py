"""Extract text lines (with a bold flag) from the Công báo PDFs.

This is the only module that imports PyMuPDF (AGPL). It runs offline at ingest time and is an
optional extra (`pip install -e ".[ingest]"`); the API and retrieval code never import it.
Its output is written to `data/interim/*.lines.jsonl` and committed, so the parser and its
tests run without PyMuPDF or the PDFs.
"""

from __future__ import annotations

import re
from pathlib import Path

from vnlaw_rag.ingest.models import Line

# Gazette pages are A4 (842 pt high). The running header "CÔNG BÁO/Số ... /Ngày ..." with the
# page number sits at y = 52..69 and the digital-signature stamp on page 1 at y < 30; body
# text starts at y >= 81 on every page (measured on all three source files).
HEADER_BAND_BOTTOM = 75.0
FOOTER_BAND_TOP = 785.0
# Body text is 14 pt (13 pt in some headings). Footnotes of the consolidated text (VBHN) are
# 12 pt, and their reference markers are 8-9 pt superscript digits.
FOOTNOTE_MAX_SIZE = 12.5
MARKER_SIZE_GAP = 3.0
BOLD_FLAG = 16  # PyMuPDF span flag bit for bold
_PAGE_NUMBER = re.compile(r"\d{1,4}")
_SPACES = re.compile(r"\s+")


def extract_lines(pdf_path: Path, part: int = 0) -> list[Line]:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            'PyMuPDF is needed to read PDFs. Install the extra: pip install -e ".[ingest]"'
        ) from exc

    lines: list[Line] = []
    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            in_footnote = False
            for block in page.get_text("dict")["blocks"]:
                for raw in block.get("lines", []):
                    found = _to_line(raw, page_index + 1, part)
                    if found is None:
                        continue
                    line, small, opens_with_marker = found
                    # A footnote opens with its superscript number on a small-font line and
                    # continues over the following small-font lines of the page. Other
                    # small text (formula cells in a table, for example) stays body text.
                    in_footnote = small and (opens_with_marker or in_footnote)
                    line.footnote = in_footnote
                    lines.append(line)
    return lines


def _to_line(raw: dict, page: int, part: int) -> tuple[Line, bool, bool] | None:
    """Return (line, small font, opens with a footnote marker), or None to drop the line."""
    spans = raw["spans"]
    visible = [s for s in spans if s["text"].strip()]
    if not visible:
        return None
    _, y0, _, y1 = raw["bbox"]
    if y1 < HEADER_BAND_BOTTOM:
        return None  # running header, page number, signature stamp
    main_size = max(s["size"] for s in visible)
    # Footnote reference markers are digits set well below the line's font size
    # (superscript): "Điều 154 ... 8a.²" or the "²" that opens a footnote.
    markers = [s for s in visible if _is_marker(s, main_size)]
    text = _SPACES.sub(" ", "".join(s["text"] for s in spans if s not in markers)).strip()
    if not text:
        return None
    if y0 > FOOTER_BAND_TOP and _PAGE_NUMBER.fullmatch(text):
        return None  # bottom page number (only on some appendix pages)
    kept = [s for s in visible if s not in markers]
    bold_chars = sum(len(s["text"].strip()) for s in kept if _is_bold(s))
    total_chars = sum(len(s["text"].strip()) for s in kept)
    # Some lines start with an empty span at the left margin; measure the first real glyph.
    first = visible[0]
    line = Line(
        page=page,
        bold=bold_chars >= 0.6 * total_chars,
        x0=round(first["bbox"][0]),
        text=text,
        part=part,
        markers=[s["text"].strip() for s in markers],
    )
    return line, main_size < FOOTNOTE_MAX_SIZE, bool(markers) and visible[0] is markers[0]


def _is_marker(span: dict, main_size: float) -> bool:
    return span["text"].strip().isdigit() and span["size"] <= main_size - MARKER_SIZE_GAP


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & BOLD_FLAG) or "Bold" in span["font"]
