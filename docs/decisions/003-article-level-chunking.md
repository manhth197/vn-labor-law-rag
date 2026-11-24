# 003. Article-level chunks with a context header, split by clause only when needed

Date: 2026-11-24 · Status: accepted

## Context
Legal questions are answered by an article or a clause, and answers must cite them. Fixed-size chunks (e.g. 512 tokens with overlap) would cut through clauses and mix neighbouring articles, which breaks citations. Article length varies a lot: measured with the bge-m3 tokenizer, the median is ~175 tokens (Labor Code) and ~300 (decree); 5 of 335 articles exceed 1,200 tokens.

## Decision
- One chunk per Điều. If the article is longer than `CHUNK_MAX_TOKENS` (1,200), split it at clause boundaries, packing consecutive clauses greedily. A single clause that is still too long is split at its points (ids `-p1`, `-p2`).
- Every chunk starts with a context header and the heading, e.g. `Bộ luật Lao động 2019 (45/2019/QH14) - Chương III. HỢP ĐỒNG LAO ĐỘNG - Mục 1. ...` then `Điều 25. Thời gian thử việc (khoản 1-2)`. Split parts repeat the article's lead-in text, because clauses like "1. Không quá 180 ngày ..." are meaningless without it.
- Token counts use the embedding model's own tokenizer (bge-m3), so the limit is exact, not estimated.
- Chunk ids are stable and readable: `BLLD2019-D25`, `ND145-D8-K1-3`, `BLLD2019-D219-K1-p2`, `ND145-PLII`. Metadata carries document, chapter, section, article, clause numbers and the official URL.

## Alternatives
- Fixed-size or recursive character splitting: simpler but ignores legal units.
- One chunk per clause: precise, but many tiny chunks lose the article context and add noise to keyword search.
- No splitting at all: bge-m3 accepts 8,192 tokens, but one vector for a 2,000-token article dilutes its meaning.

## Consequences
- 348 chunks (335 articles, 5 split into 12 parts, plus 6 appendices); the largest is 1,187 tokens.
- 1,200 is a starting point from the architecture. *Update (M6): smaller limits were not compared; it remains an open experiment, after the reranker.*
- Parsing needs the tokenizer file (downloaded once from Hugging Face); tests use a word counter.
