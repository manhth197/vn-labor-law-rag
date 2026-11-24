# 005. Default retrieval: vector-only; hybrid kept as a measured option

Date: 2026-11-24 · Status: accepted (approved by the coordinator) · Evidence: `docs/EVAL.md`

## Context
The architecture planned hybrid retrieval: bge-m3 vector search + Postgres full-text search, fused with RRF. Postgres has no Vietnamese word segmenter or stop words, so the index holds syllables. We built the keyword leg in steps and measured each one on the draft golden set (77 questions, 67 answerable, not yet reviewed by the owner):

| Configuration | Hit@5 | MRR@10 |
|---|---|---|
| keyword: OR of syllable pairs, `ts_rank_cd` | 0.313 | 0.173 |
| keyword: same terms, BM25-style IDF computed in SQL | 0.731 | 0.615 |
| vector only (bge-m3) | 0.955 | 0.858 |
| hybrid RRF (pairs + IDF + function words dropped) | 0.940 | 0.768 |

- `ts_rank_cd` has no notion of term rarity, so phrases found in almost every chunk ("người lao động", "công ty") dominate. Weighting each term by IDF, ln(1 + (N − df + 0.5)/(df + 0.5)) with df counted through the GIN index at query time, more than doubled keyword Hit@5. Syllable pairs beat single syllables (hybrid Hit@5 0.940 vs 0.881).
- Even so, equal-weight RRF does not beat vector-only. By type, hybrid helps only numeric questions (Hit@5 1.00 vs 0.96, one question) and hurts paraphrased everyday questions (MRR 0.51 vs 0.65), which share few words with the legal text.

## Decision
`RETRIEVAL_MODE=vector` by default. The keyword leg, the IDF scoring and hybrid RRF stay in the code, in the tests and in every eval run, selectable with `RETRIEVAL_MODE=hybrid`. We did not tune RRF weights or the stop list on this set: it is small, unreviewed and written by the same agent that wrote the retriever, so tuning would overfit.

## Alternatives
- Keep hybrid because the architecture said so: contradicts our own measurement.
- Weighted RRF or a learned fusion weight: needs a reviewed, larger set (ideally with keyword-heavy questions) to tune without overfitting.

## Consequences
- Simpler request path for M3 (one embedding call + one SQL query); explicit article references ("Điều 35") are handled by the router's regex lookup, not by keyword search.
- Revisit after the owner reviews the golden set: add keyword-heavy questions, then compare weighted RRF and the bge-reranker-v2-m3 cross-encoder (M6).
