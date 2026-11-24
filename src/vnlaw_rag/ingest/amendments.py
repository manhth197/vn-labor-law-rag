"""Registry of amendments to the corpus, with effective dates (`data/amendments.json`).

A consolidated text (VBHN) lags months behind the laws that amend it, and an amendment can be
passed long before it takes effect. The registry records each amending document with its
status, effective date, the articles it touches and an official source, so every chunk can
say what already changed (`amended_by`) and what will change (`upcoming_change`).

- `in_corpus_text`: the corpus text already contains the change (e.g. through the VBHN).
- `mapping`: "explicit" when the article list comes from the source text (a VBHN footnote),
  "inferred_from_topic" when the official source only names the topic and the articles are
  the ones that regulate that topic.
- `use_in_answers: false` (drafts) keeps an entry for reference only.
See docs/decisions/006-consolidated-text-and-amendment-registry.md.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from vnlaw_rag.ingest.models import Chunk

Status = Literal["in_force", "passed_pending_text", "draft"]
Mapping = Literal["explicit", "inferred_from_topic"]


class AmendingDocument(BaseModel):
    number: str | None = None  # None until the official text is published
    title: str
    date: str  # date passed (law) or signed/published (decree, draft); ISO, may be YYYY-MM


class Effect(BaseModel):
    doc: str
    articles: list[int]
    clauses: list[str] = Field(default_factory=list)  # empty: the article in general
    change: str  # user-facing, Vietnamese
    mapping: Mapping


class Source(BaseModel):
    url: str
    title: str
    accessed: str


class Amendment(BaseModel):
    id: str
    amending: AmendingDocument
    status: Status
    effective_date: str | None = None
    in_corpus_text: bool
    use_in_answers: bool
    summary: str
    affects: list[Effect] = Field(default_factory=list)
    sources: list[Source]


class Registry(BaseModel):
    checked_at: str  # date the registry was last checked against official sources
    amendments: list[Amendment]


def load_registry(path: Path) -> Registry:
    return Registry.model_validate_json(path.read_text(encoding="utf-8"))


def vn_date(iso: str) -> str:
    parts = iso.split("-")
    return "/".join(reversed(parts))


def _label(doc: AmendingDocument) -> str:
    return f"{doc.title} số {doc.number}" if doc.number else doc.title


def annotate_chunks(chunks: Iterable[Chunk], registry: Registry) -> None:
    """Fill `amended_by` and `upcoming_change` on article chunks, in place."""
    for chunk in chunks:
        if chunk.article_no is None:
            continue
        for amendment in registry.amendments:
            if not amendment.use_in_answers:
                continue
            for effect in amendment.affects:
                if effect.doc != chunk.doc_short or chunk.article_no not in effect.articles:
                    continue
                # A chunk holding only some clauses gets notes about those clauses only.
                partial = effect.clauses and chunk.clause_nos
                if partial and not set(effect.clauses) & set(chunk.clause_nos):
                    continue
                message = _message(amendment, effect, registry.checked_at)
                target = chunk.amended_by if amendment.in_corpus_text else chunk.upcoming_change
                if message not in target:
                    target.append(message)


def _message(amendment: Amendment, effect: Effect, checked_at: str) -> str:
    source = _label(amendment.amending)
    effective = vn_date(amendment.effective_date) if amendment.effective_date else "chưa rõ"
    hedge = " (có thể bị ảnh hưởng; chưa có toàn văn)" if effect.mapping != "explicit" else ""
    if amendment.in_corpus_text:
        return f"{effect.change}. Theo {source}, hiệu lực từ {effective}."
    if amendment.effective_date and amendment.effective_date <= checked_at:
        return (
            f"Đã thay đổi từ {effective}, văn bản trong hệ thống chưa cập nhật: "
            f"{effect.change}{hedge}. Theo {source}."
        )
    return (
        f"Sẽ thay đổi từ {effective}: {effect.change}{hedge}. "
        f"Theo {source} (thông qua {vn_date(amendment.amending.date)})."
    )


def corpus_notice(registry: Registry, text_versions: Iterable[str]) -> str:
    """One-line notice for answers and the UI, e.g. "Theo VBHN 125/VBHN-VPQH (2025); một số
    quy định sẽ thay đổi từ 01/03/2027."."""
    notice = "Theo " + ", ".join(text_versions)
    upcoming = sorted(
        a.effective_date
        for a in registry.amendments
        if a.use_in_answers
        and not a.in_corpus_text
        and a.effective_date
        and a.effective_date > registry.checked_at
    )
    if upcoming:
        notice += f"; một số quy định sẽ thay đổi từ {vn_date(upcoming[0])}"
    return notice + "."
