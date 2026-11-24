"""Turn PDF lines into the legal structure: Chương > Mục > Điều > Khoản > Điểm (+ Phụ lục).

The parser relies on four signals measured on the Công báo layout, in this order:

1. **Bold** marks headings: "Chương", "Mục", "Điều N." and their (possibly wrapped) titles.
2. **Indentation** marks the start of a paragraph. Body text has a 64 pt left margin and a
   first-line indent at 87 pt, so a clause "2." at the start of a wrapped line is just text.
3. **Numbering must be sequential**: a clause is accepted only if it is the next number in its
   article, a point only if it is the next letter (a, b, c, d, đ, e, g, ...). Anything else is
   kept as text and recorded as an anomaly for review.
4. **Quotes**: amending articles quote other laws ("“Điều 54. ...”"); text inside “...” is
   never structure.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from vnlaw_rag.ingest.models import (
    Anomaly,
    Appendix,
    Article,
    Clause,
    Line,
    ParsedDocument,
    Point,
)
from vnlaw_rag.ingest.sources import SourceDocument

INDENT_MIN_X = 75  # left margin is 64 pt, first-line indent 87 pt

CHAPTER_RE = re.compile(r"^Chương ([IVXLC]+)$")
SECTION_RE = re.compile(r"^Mục (\d+)$")
ARTICLE_RE = re.compile(r"^Điều (\d+)\.\s*(.*)$")
CLAUSE_RE = re.compile(r"^(\d+)\.\s+(.*)$")
POINT_RE = re.compile(r"^([a-zđ])\)\s+(.*)$")
APPENDIX_RE = re.compile(r"^Phụ lục ([IVXLC]+)$")
FORM_ENTRY_RE = re.compile(r"^Mẫu số (\d+/PL[IVXLC]+)\s*(.*)$")
STRUCTURAL = (CHAPTER_RE, SECTION_RE, ARTICLE_RE)

# Vietnamese legal texts letter their points with the Vietnamese alphabet, skipping f, j, w, z.
POINT_LETTERS = "abcdđeghiklmnopqrstuvxy"
OPEN_QUOTE, CLOSE_QUOTE = "“", "”"


class ParseError(ValueError):
    pass


def parse_document(lines: list[Line], doc: SourceDocument) -> ParsedDocument:
    body, tail = split_body(lines, doc)
    builder = _ArticleBuilder(doc)
    builder.feed(body)
    appendices = parse_appendices(tail, doc)
    return ParsedDocument(
        doc_short=doc.doc_short,
        articles=builder.articles,
        appendices=appendices,
        anomalies=builder.anomalies,
    )


def split_body(lines: list[Line], doc: SourceDocument) -> tuple[list[Line], list[Line]]:
    """Return (normative body, everything after the signature block).

    Each source file starts with front matter (gazette banner, national motto, preamble or a
    "continued from issue ..." note); it is skipped up to the first bold structural heading.
    """
    end_patterns = [re.compile(p) for p in doc.body_end_patterns]
    body: list[Line] = []
    tail: list[Line] = []
    ended = False
    for part in sorted({line.part for line in lines}):
        part_lines = [line for line in lines if line.part == part]
        start = next(
            (i for i, line in enumerate(part_lines) if line.bold and _is_structural(line.text)),
            None,
        )
        if start is None:
            if not ended:
                raise ParseError(f"{doc.doc_short} part {part}: no structural heading found")
            start = 0
        for line in part_lines[start if not ended else 0 :]:
            if not ended and any(p.match(line.text) for p in end_patterns):
                ended = True
            (tail if ended else body).append(line)
    if not ended:
        raise ParseError(f"{doc.doc_short}: end of body (signature block) not found")
    return body, tail


def _is_structural(text: str) -> bool:
    return any(p.match(text) for p in STRUCTURAL)


def _join(parts: Iterable[str]) -> str:
    return " ".join(p.strip() for p in parts if p.strip())


class _ArticleBuilder:
    def __init__(self, doc: SourceDocument) -> None:
        self.doc = doc
        self.articles: list[Article] = []
        self.anomalies: list[Anomaly] = []
        self.chapter_seq = 0
        self.chapter_no = ""
        self.chapter_title = ""
        self.section_no: int | None = None
        self.section_title: str | None = None
        self.quote_depth = 0

    # -- entry point -------------------------------------------------------------------
    def feed(self, lines: list[Line]) -> None:
        i = 0
        while i < len(lines):
            line = lines[i]
            in_quote = self.quote_depth > 0 or line.text.startswith(OPEN_QUOTE)
            if line.bold and not in_quote:
                consumed = self._heading(lines, i)
                if consumed:
                    i += consumed
                    continue
                self._anomaly(line, "bold line that is not a heading; kept as text")
            self._text(line, in_quote)
            self.quote_depth += line.text.count(OPEN_QUOTE) - line.text.count(CLOSE_QUOTE)
            self.quote_depth = max(self.quote_depth, 0)
            i += 1
        if self.quote_depth:
            raise ParseError(f"{self.doc.doc_short}: unbalanced quotes at end of body")

    # -- headings ----------------------------------------------------------------------
    def _heading(self, lines: list[Line], i: int) -> int:
        """Handle a heading at lines[i]; return how many lines it used (0 if not a heading)."""
        text = lines[i].text
        if m := CHAPTER_RE.match(text):
            title, n = self._bold_continuation(lines, i + 1)
            if (m[1], title) != (self.chapter_no, self.chapter_title):
                # A gazette issue that continues a document repeats the current chapter
                # heading; only a different (number, title) pair is a new chapter. The
                # number alone is not enough: Nghị định 145 prints "Chương XI" twice.
                self.chapter_seq += 1
                self.chapter_no, self.chapter_title = m[1], title
                self.section_no = self.section_title = None
            return 1 + n
        if m := SECTION_RE.match(text):
            title, n = self._bold_continuation(lines, i + 1)
            self.section_no, self.section_title = int(m[1]), title
            return 1 + n
        if m := ARTICLE_RE.match(text):
            number = int(m[1])
            expected = self.articles[-1].article_no + 1 if self.articles else 1
            if number != expected:
                raise ParseError(
                    f"{self.doc.doc_short}: expected Điều {expected}, found {text!r} "
                    f"(part {lines[i].part}, page {lines[i].page})"
                )
            rest, n = self._bold_continuation(lines, i + 1)
            self.articles.append(
                Article(
                    doc_short=self.doc.doc_short,
                    doc_number=self.doc.doc_number,
                    article_no=number,
                    article_title=_join([m[2], rest]),
                    chapter_seq=self.chapter_seq,
                    chapter_no=self.chapter_no,
                    chapter_title=self.chapter_title,
                    section_no=self.section_no,
                    section_title=self.section_title,
                    source_url=self.doc.source_url,
                    pages=[lines[i].page],
                )
            )
            return 1 + n
        return 0

    @staticmethod
    def _bold_continuation(lines: list[Line], j: int) -> tuple[str, int]:
        """Collect the bold lines that continue a heading's title (titles can wrap)."""
        parts: list[str] = []
        while j + len(parts) < len(lines):
            nxt = lines[j + len(parts)]
            if not nxt.bold or _is_structural(nxt.text) or nxt.text.startswith(OPEN_QUOTE):
                break
            parts.append(nxt.text)
        return _join(parts), len(parts)

    # -- body text ---------------------------------------------------------------------
    def _text(self, line: Line, in_quote: bool) -> None:
        if not self.articles:
            self._anomaly(line, "text before the first article; dropped")
            return
        article = self.articles[-1]
        if line.page not in article.pages:
            article.pages.append(line.page)
        text = line.text
        indented = line.x0 >= INDENT_MIN_X

        if indented and not in_quote:
            if m := CLAUSE_RE.match(text):
                expected = article.clauses[-1].no + 1 if article.clauses else 1
                if int(m[1]) == expected:
                    article.clauses.append(Clause(no=expected, text=m[2].strip()))
                    return
                self._anomaly(line, f"looks like clause {m[1]} but expected {expected}")
            elif m := POINT_RE.match(text):
                points = article.clauses[-1].points if article.clauses else article.intro_points
                expected_letter = _next_letter(points[-1].letter if points else None)
                if m[1] == expected_letter:
                    points.append(Point(letter=m[1], text=m[2].strip()))
                    return
                self._anomaly(line, f"looks like point {m[1]}) but expected {expected_letter})")

        self._append(article, text, new_paragraph=indented)

    @staticmethod
    def _append(article: Article, text: str, *, new_paragraph: bool) -> None:
        sep = "\n" if new_paragraph else " "
        if article.clauses:
            clause = article.clauses[-1]
            if clause.points:
                clause.points[-1].text = _concat(clause.points[-1].text, text, sep)
            else:
                clause.text = _concat(clause.text, text, sep)
        elif article.intro_points:
            article.intro_points[-1].text = _concat(article.intro_points[-1].text, text, sep)
        else:
            article.intro = _concat(article.intro, text, sep)

    def _anomaly(self, line: Line, message: str) -> None:
        self.anomalies.append(
            Anomaly(
                doc_short=self.doc.doc_short,
                article_no=self.articles[-1].article_no if self.articles else None,
                page=line.page,
                part=line.part,
                message=message,
                text=line.text,
            )
        )


def _concat(existing: str, text: str, sep: str) -> str:
    return f"{existing}{sep}{text}" if existing else text


def _next_letter(letter: str | None) -> str:
    if letter is None:
        return POINT_LETTERS[0]
    index = POINT_LETTERS.find(letter)
    return POINT_LETTERS[index + 1] if 0 <= index < len(POINT_LETTERS) - 1 else "?"


# -- appendices --------------------------------------------------------------------------
def parse_appendices(tail: list[Line], doc: SourceDocument) -> list[Appendix]:
    """Split the lines after the signature block into appendices (Phụ lục).

    Substantive lists are kept in full. Form collections ("Mẫu số 01/PLIII", ...) keep only
    their table of contents: the forms themselves are templates, not rules.
    """
    starts = [i for i, line in enumerate(tail) if line.bold and APPENDIX_RE.match(line.text)]
    appendices: list[Appendix] = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(tail)
        block = tail[start:end]
        number = APPENDIX_RE.match(block[0].text)[1]
        title_lines: list[str] = []
        j = 1
        while j < len(block) and block[j].bold and not block[j].text.startswith("("):
            title_lines.append(block[j].text)
            j += 1
        j = _skip_attachment_note(block, j)
        content = block[j:]
        if content and FORM_ENTRY_RE.match(content[0].text):
            # Form collections: the first page is a table of contents; forms start on the
            # next page. Keep only the contents.
            first_page = (content[0].part, content[0].page)
            text = _forms_table_of_contents(
                [line for line in content if (line.part, line.page) == first_page]
            )
            kind = "forms"
            title = _join(title_lines) or "Biểu mẫu"
        else:
            text = _list_text(content)
            kind = "list"
            title = _join(title_lines)
        appendices.append(
            Appendix(
                doc_short=doc.doc_short,
                doc_number=doc.doc_number,
                appendix_no=number,
                title=title,
                kind=kind,
                text=text,
                source_url=doc.source_url,
            )
        )
    return appendices


def _skip_attachment_note(block: list[Line], j: int) -> int:
    """Skip "(Kèm theo Nghị định số ... của Chính phủ)" and separator rules."""
    if j < len(block) and block[j].text.startswith("(Kèm theo"):
        while j < len(block) and not block[j].text.endswith(")"):
            j += 1
        j += 1
    while j < len(block) and set(block[j].text) <= {"_", "-", " "}:
        j += 1
    return j


def _forms_table_of_contents(lines: list[Line]) -> str:
    """One "Mẫu số NN/PLx: name" entry per form; wrapped names are joined."""
    entries: list[str] = []
    for line in lines:
        if m := FORM_ENTRY_RE.match(line.text):
            entries.append(_join([f"Mẫu số {m[1]}:", m[2]]))
        elif entries:
            entries[-1] = _join([entries[-1], line.text])
    return "\n".join(entries)


def _list_text(content: list[Line]) -> str:
    """Join list lines into one item per line.

    Table layout (a column of bare row numbers, "STT"): a row is its number plus every line
    up to the next number; the header row before the first number is dropped. Otherwise a
    paragraph starts at an indented or bold line and wrapped lines are joined to it.
    """
    items: list[str] = []
    if any(line.text.isdigit() for line in content):
        for line in content:
            if line.text.isdigit():
                items.append(f"{line.text}.")
            elif items:
                items[-1] = _join([items[-1], line.text])
        return "\n".join(items)
    prev: Line | None = None
    for line in content:
        wrapped_heading = line.bold and prev is not None and prev.bold
        if not items or (not wrapped_heading and (line.x0 >= INDENT_MIN_X or line.bold)):
            items.append(line.text)
        else:
            items[-1] = _join([items[-1], line.text])
        prev = line
    return "\n".join(items)
