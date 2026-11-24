"""Split parsed articles into retrieval chunks.

Rule (docs/ARCHITECTURE.md): one chunk per Điều. An article longer than `max_tokens` is split
at clause (Khoản) boundaries, packing consecutive clauses greedily; a single clause that is
still too long is split at point (Điểm) boundaries. Every part repeats the context header, the
article heading and the article's lead-in text, so a part is understandable on its own
("Thời gian thử việc ... bảo đảm điều kiện sau đây:" gives meaning to "1. Không quá 180 ngày").

Token counts use the embedding model's own tokenizer (bge-m3), so the budget is exact.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vnlaw_rag.ingest.models import Appendix, Article, Chunk, Clause, Point, clause_text
from vnlaw_rag.ingest.sources import SourceDocument, get_document

TokenCounter = Callable[[str], int]


def load_token_counter(tokenizer_name: str) -> TokenCounter:
    """Token counter backed by a Hugging Face tokenizer (downloads tokenizer.json once)."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_pretrained(tokenizer_name)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


def context_header(doc: SourceDocument, article: Article) -> str:
    parts = [
        f"{doc.short_title} ({doc.doc_number})",
        f"Chương {article.chapter_no}. {article.chapter_title}",
    ]
    if article.section_no is not None:
        parts.append(f"Mục {article.section_no}. {article.section_title}")
    return " - ".join(parts)


@dataclass
class _Unit:
    """A piece of an article body that can be packed into a chunk."""

    clause_no: str  # clause label, e.g. "8a"
    text: str
    tokens: int
    piece: bool = False  # one of several pieces of a clause that was split at its points


def chunk_article(article: Article, count_tokens: TokenCounter, max_tokens: int) -> list[Chunk]:
    doc = get_document(article.doc_short)
    header = context_header(doc, article)
    whole = _compose(header, article.heading, article.body_text())
    if count_tokens(whole) <= max_tokens or not article.clauses:
        # Articles without clauses are never split: they are short in this corpus, and an
        # oversized one would still fit bge-m3's 8,192-token window.
        return [_make_chunk(doc, article, whole, [], count_tokens)]

    lead = article.intro  # lead-in repeated in every part
    lead_tokens = count_tokens(_compose(header, article.heading + " (khoản 99-99)", lead))
    budget = max_tokens - lead_tokens
    units = _clause_units(article.clauses, count_tokens, budget)

    chunks: list[Chunk] = []
    for group in _pack(units, budget):
        clause_nos = list(dict.fromkeys(u.clause_no for u in group))  # keep text order
        body = "\n".join(filter(None, [lead, *(u.text for u in group)]))
        heading = f"{article.heading} ({_clause_label(clause_nos)})"
        text = _compose(header, heading, body)
        chunks.append(_make_chunk(doc, article, text, clause_nos, count_tokens))
    _number_repeated_ids(chunks)
    return chunks


def _clause_units(clauses: list[Clause], count_tokens: TokenCounter, budget: int) -> list[_Unit]:
    units: list[_Unit] = []
    for clause in clauses:
        text = clause_text(clause)
        tokens = count_tokens(text)
        if tokens <= budget or not clause.points:
            units.append(_Unit(clause.label, text, tokens))
            continue
        # Too long on its own: split at point boundaries, repeating the clause's lead text.
        lead = f"{clause.label}. {clause.text}".rstrip()
        lead_tokens = count_tokens(lead)
        groups = _pack(
            [
                _Unit(clause.label, _point_text(p), count_tokens(_point_text(p)))
                for p in clause.points
            ],
            budget - lead_tokens,
        )
        for group in groups:
            text = "\n".join([lead, *(u.text for u in group)])
            units.append(_Unit(clause.label, text, count_tokens(text), piece=True))
    return units


def _point_text(point: Point) -> str:
    return f"{point.letter}) {point.text}"


def _pack(units: list[_Unit], budget: int) -> list[list[_Unit]]:
    """Greedy packing of consecutive units; a unit larger than the budget stays alone.
    A piece of a split clause always gets its own chunk, so chunk ids and the "(khoản N)"
    label never mix a partial clause with whole ones."""
    groups: list[list[_Unit]] = []
    used = 0
    for unit in units:
        fits = groups and used + unit.tokens + 1 <= budget
        if fits and not unit.piece and not groups[-1][-1].piece:
            groups[-1].append(unit)
            used += unit.tokens + 1
        else:
            groups.append([unit])
            used = unit.tokens
    return groups


def _number_repeated_ids(chunks: list[Chunk]) -> None:
    """Pieces of one split clause get the same base id; suffix them with -p1, -p2, ..."""
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.chunk_id] = counts.get(c.chunk_id, 0) + 1
    seen: dict[str, int] = {}
    for c in chunks:
        if counts[c.chunk_id] > 1:
            seen[c.chunk_id] = seen.get(c.chunk_id, 0) + 1
            c.chunk_id = f"{c.chunk_id}-p{seen[c.chunk_id]}"


def _clause_label(clause_nos: list[str]) -> str:
    if len(clause_nos) == 1:
        return f"khoản {clause_nos[0]}"
    return f"khoản {clause_nos[0]}-{clause_nos[-1]}"


def _compose(header: str, heading: str, body: str) -> str:
    return "\n".join(filter(None, [header, heading, body]))


def _make_chunk(
    doc: SourceDocument,
    article: Article,
    text: str,
    clause_nos: list[str],
    count_tokens: TokenCounter,
) -> Chunk:
    chunk_id = f"{doc.doc_short}-D{article.article_no}"
    if clause_nos:
        chunk_id += f"-K{clause_nos[0]}" + (f"-{clause_nos[-1]}" if len(clause_nos) > 1 else "")
    return Chunk(
        chunk_id=chunk_id,
        doc_short=doc.doc_short,
        doc_number=doc.doc_number,
        doc_title=doc.short_title,
        unit="article",
        chapter_no=article.chapter_no,
        chapter_title=article.chapter_title,
        section_no=article.section_no,
        section_title=article.section_title,
        article_no=article.article_no,
        article_title=article.article_title,
        clause_nos=clause_nos,
        source_url=article.source_url,
        text=text,
        n_tokens=count_tokens(text),
    )


def chunk_appendix(appendix: Appendix, count_tokens: TokenCounter) -> Chunk:
    doc = get_document(appendix.doc_short)
    heading = f"Phụ lục {appendix.appendix_no}. {appendix.title}"
    text = _compose(f"{doc.short_title} ({doc.doc_number})", heading, appendix.text)
    return Chunk(
        chunk_id=f"{doc.doc_short}-PL{appendix.appendix_no}",
        doc_short=doc.doc_short,
        doc_number=doc.doc_number,
        doc_title=doc.short_title,
        unit="appendix",
        appendix_no=appendix.appendix_no,
        source_url=appendix.source_url,
        text=text,
        n_tokens=count_tokens(text),
    )


def build_chunks(
    articles: Iterable[Article],
    appendices: Iterable[Appendix],
    count_tokens: TokenCounter,
    max_tokens: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(chunk_article(article, count_tokens, max_tokens))
    chunks.extend(chunk_appendix(a, count_tokens) for a in appendices)
    ids = [c.chunk_id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk ids")
    return chunks
