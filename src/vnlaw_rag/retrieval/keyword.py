"""Build Postgres full-text query terms from a Vietnamese question.

Vietnamese words are mostly two syllables written with a space ("thử việc", "lao động"), and
Postgres' `simple` parser splits on spaces, so the index holds syllables. Two term shapes:

- "unigram": single syllables. High recall, but common syllables ("người", "lao", "động")
  match almost every chunk.
- "bigram" (default): adjacent syllable pairs as phrases (`thu <-> viec`), which approximates
  matching whole words such as "thử việc" without a word segmenter.

Terms are OR-ed (a question never contains all the words of the passage that answers it, so
AND semantics returns nothing for most natural questions). Optionally, terms that contain a
Vietnamese function word ("có", "không", "bao nhiêu", "tôi", ...) are dropped: the list is a
generic stop list, not tuned on the golden set. Terms are letters and digits only, so they
cannot inject tsquery operators. Accents are removed by the `vn_unaccent` config in Postgres.
"""

from __future__ import annotations

import re
from typing import Literal

KeywordMode = Literal["unigram", "bigram"]

_SYLLABLE = re.compile(r"[^\W_]+")

# Function words and question words of everyday Vietnamese (pronouns, particles,
# interrogatives, copula, common connectives). Generic; not derived from the golden set.
STOP_SYLLABLES = frozenset(
    """
    tôi ta mình bạn anh chị em họ nó chúng
    có không được bị thì là mà và hoặc hay với cho của này đó kia ấy
    gì nào sao bao nhiêu thế như vậy đâu ai mấy
    các những mọi một rồi đã đang sẽ cũng vẫn chỉ nếu khi để vì do
    ạ nhé hả à ơi nhỉ chứ
    """.split()  # noqa: SIM905  (readable as prose)
)


def syllables(text: str) -> list[str]:
    return [s.lower() for s in _SYLLABLE.findall(text)]


def query_terms(
    question: str, mode: KeywordMode = "bigram", drop_stopwords: bool = False
) -> list[str]:
    """Return unique tsquery terms in question order, e.g. ["thử <-> việc", ...]."""
    tokens = syllables(question)
    stop = STOP_SYLLABLES if drop_stopwords else frozenset()
    if mode == "unigram" or len(tokens) < 2:
        terms = [t for t in tokens if t not in stop]
    elif mode == "bigram":
        terms = [
            f"{a} <-> {b}"
            for a, b in zip(tokens, tokens[1:], strict=False)
            if a not in stop and b not in stop
        ]
    else:
        raise ValueError(f"unknown keyword mode {mode!r}")
    return list(dict.fromkeys(terms))  # dedupe, keep order


def build_tsquery(question: str, mode: KeywordMode = "bigram", drop_stopwords: bool = False) -> str:
    """All terms OR-ed into one tsquery string."""
    return " | ".join(query_terms(question, mode, drop_stopwords))
