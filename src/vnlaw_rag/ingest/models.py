"""Data models for the ingestion pipeline: PDF lines, the legal structure, and chunks."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Line(BaseModel):
    """One visual text line extracted from a PDF page (page headers already removed)."""

    page: int  # 1-based page number inside its source file
    bold: bool
    x0: int  # left edge of the first visible glyph, in PDF points
    text: str  # footnote reference markers removed
    part: int  # index of the source file (documents can span several gazette issues)
    footnote: bool = False  # a footnote line (smaller font, page bottom), not body text
    markers: list[str] = Field(default_factory=list)  # superscript footnote numbers on the line


class Point(BaseModel):
    """Điểm: a lettered item, e.g. `a)`."""

    letter: str
    text: str


class Clause(BaseModel):
    """Khoản: a numbered item, e.g. `1.`; amendments insert lettered ones such as `8a.`."""

    no: int
    suffix: str = ""  # "a" for an inserted clause "8a"
    text: str  # the clause's own lead text (before its points)
    points: list[Point] = Field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.no}{self.suffix}"


class SourceNote(BaseModel):
    """A footnote of the source text attached to an article (consolidated texts use them to
    say which law amended which clause)."""

    clause: str | None = None  # clause label the note is attached to, None for the article
    text: str


class Article(BaseModel):
    """Điều: the main citable unit. One JSONL record per article."""

    doc_short: str
    doc_number: str
    article_no: int
    article_title: str
    chapter_seq: int  # 1-based order of the chapter in the document (numbering can repeat)
    chapter_no: str  # roman numeral exactly as printed, e.g. "III"
    chapter_title: str
    section_no: int | None = None  # Mục
    section_title: str | None = None
    intro: str = ""  # text before the first clause (or the whole body if no clauses)
    intro_points: list[Point] = Field(default_factory=list)  # points with no parent clause
    clauses: list[Clause] = Field(default_factory=list)
    notes: list[SourceNote] = Field(default_factory=list)
    source_url: str
    pages: list[int] = Field(default_factory=list)  # gazette pages (file-local) it spans

    @property
    def heading(self) -> str:
        return f"Điều {self.article_no}. {self.article_title}"

    def body_text(self) -> str:
        """Plain text of the article body, one structural item per line."""
        lines: list[str] = []
        if self.intro:
            lines.append(self.intro)
        lines.extend(f"{p.letter}) {p.text}" for p in self.intro_points)
        for clause in self.clauses:
            lines.append(clause_text(clause))
        return "\n".join(lines)


def clause_text(clause: Clause) -> str:
    parts = [f"{clause.label}. {clause.text}".rstrip()]
    parts.extend(f"{p.letter}) {p.text}" for p in clause.points)
    return "\n".join(parts)


class Appendix(BaseModel):
    """Phụ lục. `kind` is "list" for substantive lists, "forms" for template collections
    (only their table of contents is kept)."""

    doc_short: str
    doc_number: str
    appendix_no: str  # roman numeral as printed
    title: str
    kind: str
    text: str
    source_url: str


class Anomaly(BaseModel):
    """Something the parser noticed but did not treat as structure. Reviewed, not hidden."""

    doc_short: str
    article_no: int | None
    page: int
    part: int
    message: str
    text: str


class ParsedDocument(BaseModel):
    doc_short: str
    articles: list[Article]
    appendices: list[Appendix] = Field(default_factory=list)
    anomalies: list[Anomaly] = Field(default_factory=list)
    footnotes: dict[str, str] = Field(default_factory=dict)  # all footnotes, by number


class Chunk(BaseModel):
    """A retrieval unit: a whole article, a group of its clauses when the article is too long,
    or an appendix."""

    chunk_id: str
    doc_short: str
    doc_number: str
    doc_title: str
    unit: str  # "article" | "appendix"
    chapter_no: str | None = None
    chapter_title: str | None = None
    section_no: int | None = None
    section_title: str | None = None
    article_no: int | None = None
    article_title: str | None = None
    clause_nos: list[str] = Field(default_factory=list)  # clause labels; empty = whole unit
    appendix_no: str | None = None
    source_url: str
    text: str  # what gets embedded and keyword-indexed: context header + body
    n_tokens: int
    # From data/amendments.json (see ingest/amendments.py): amendments already in the text,
    # and announced changes not yet in the text (with their effective date).
    amended_by: list[str] = Field(default_factory=list)
    upcoming_change: list[str] = Field(default_factory=list)
