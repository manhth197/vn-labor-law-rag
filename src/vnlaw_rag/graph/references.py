"""Find explicit article references in a question: "Điều 35", "khoản 2 Điều 113 BLLĐ",
"dieu 21 nghi dinh 145". The `route` node sends such questions to a direct lookup.

Matching runs on an accent-free, lower-case copy of the question so that typing without
diacritics works. "điều kiện" (condition) is not a reference: a number must follow "điều".
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_ARTICLE = re.compile(r"(?:khoan\s+(\d+[a-z]?)\s*,?\s*)?dieu\s+(\d{1,3})\b")
_DECREE = re.compile(r"nghi\s*dinh\s*(?:so\s*)?145|\bnd\s*-?\s*145\b|145\s*/\s*2020")
_CODE = re.compile(r"bo\s*luat\s*lao\s*dong|\bblld\b")


@dataclass(frozen=True)
class ArticleRef:
    doc_short: str
    article_no: int
    clause: str | None = None


def fold(text: str) -> str:
    """Lower-case, strip Vietnamese diacritics (đ -> d)."""
    text = text.lower().replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def find_references(question: str) -> list[ArticleRef]:
    """References in question order. The document is Nghị định 145 when the question names it
    (and not the Labor Code), otherwise the Labor Code."""
    folded = fold(question)
    decree = bool(_DECREE.search(folded)) and not _CODE.search(folded)
    doc = "ND145" if decree else "BLLD2019"
    refs: list[ArticleRef] = []
    for clause, article in _ARTICLE.findall(folded):
        ref = ArticleRef(doc, int(article), clause or None)
        if ref not in refs:
            refs.append(ref)
    return refs
