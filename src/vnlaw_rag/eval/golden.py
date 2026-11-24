"""Golden set: questions with expected article references, loaded and validated.

Each item lists `evidence`: short quotes copied from the expected articles. `validate_golden`
checks every quote really occurs in one of the expected units, so a wrong article number is
caught mechanically before the owner's manual review (`reviewed: false` until then).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from vnlaw_rag.ingest.models import Appendix, Article

QuestionType = Literal["lookup", "paraphrase", "multi", "numeric", "unanswerable"]
UnitKey = tuple[str, str, str]  # (doc_short, "article" | "appendix", number)


class Ref(BaseModel):
    doc: str
    article: int | None = None
    appendix: str | None = None

    def key(self) -> UnitKey:
        if self.article is not None:
            return (self.doc, "article", str(self.article))
        return (self.doc, "appendix", str(self.appendix))


class GoldenItem(BaseModel):
    id: str
    question: str
    type: QuestionType
    expected: list[Ref] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    reference_answer: str
    reviewed: bool = False
    drafted_by: str = ""
    notes: str | None = None
    review_status: str | None = None  # "ok" | "fixed" | "needs_fix" (set by review import)
    review_note: str | None = None  # the reviewer's free-text note

    @property
    def answerable(self) -> bool:
        return self.type != "unanswerable"


def load_golden(path: Path) -> list[GoldenItem]:
    with path.open(encoding="utf-8") as fh:
        return [GoldenItem.model_validate_json(row) for row in fh if row.strip()]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def unit_texts(articles: Iterable[Article], appendices: Iterable[Appendix]) -> dict[UnitKey, str]:
    texts = {
        (a.doc_short, "article", str(a.article_no)): _norm(f"{a.heading} {a.body_text()}")
        for a in articles
    }
    texts.update(
        {(p.doc_short, "appendix", p.appendix_no): _norm(f"{p.title} {p.text}") for p in appendices}
    )
    return texts


def validate_golden(items: list[GoldenItem], texts: dict[UnitKey, str]) -> list[str]:
    """Return a list of problems (empty when the set is consistent with the corpus)."""
    problems: list[str] = []
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        problems.append("duplicate ids")
    for item in items:
        if not item.answerable:
            if item.expected or item.evidence:
                problems.append(f"{item.id}: unanswerable item must not list references")
            continue
        if not item.expected or not item.evidence:
            problems.append(f"{item.id}: answerable item needs expected refs and evidence")
        unknown = [r.key() for r in item.expected if r.key() not in texts]
        if unknown:
            problems.append(f"{item.id}: unknown references {unknown}")
        for quote in item.evidence:
            if not any(_norm(quote) in texts.get(r.key(), "") for r in item.expected):
                problems.append(f"{item.id}: evidence not found in expected units: {quote!r}")
    return problems
