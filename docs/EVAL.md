# Evaluation

No invented numbers: every table below is produced by a command and records its date, git revision and configuration.

<!-- retrieval:start -->
## Retrieval

Run: 2026-11-24 · git `bda2382` · command `python -m vnlaw_rag.cli eval`

- Embeddings: `BAAI/bge-m3` (normalized, cosine, HNSW). Chunks: 348 (max 1200 tokens), chunks.jsonl sha256 `00f23290ee18`.
- Candidates: vector top 20, keyword top 20; RRF k = 60. Metrics at depth 10, article-level matching.
- Golden set: `data/eval/golden.jsonl`, 77 questions (numeric 23, lookup 15, paraphrase 19, multi 10, unanswerable 10). **Reviewed by the owner: 0/77.** Unanswerable questions are excluded from the ranking metrics.

> **Caveat (golden set chưa duyệt, do agent soạn, có thể thiên lệch).** The questions were drafted by the coding agent from the article texts, not by users. Lookup items reuse the law's wording, and there are few keyword-heavy questions (exact terms, document numbers). Treat the numbers as a development signal until the owner's review; they may move after it.

Configuration names: `keyword-bigram-tsrank` = OR of syllable pairs ranked by `ts_rank_cd`; `-idf` = BM25-style IDF weighting computed in SQL; `-stop` = terms with Vietnamese function words dropped; `hybrid-*` = that keyword leg fused with the vector leg by RRF.

### Overall (answerable questions)

| Configuration | Hit@1 | Hit@5 | MRR@10 | Recall@10 |
|---|---|---|---|---|
| **vector** (default) | 0.776 | 0.955 | 0.858 | 0.970 |
| keyword-bigram-tsrank | 0.075 | 0.313 | 0.173 | 0.455 |
| keyword-bigram-idf | 0.537 | 0.731 | 0.615 | 0.806 |
| keyword-bigram-idf-stop | 0.388 | 0.761 | 0.545 | 0.811 |
| keyword-unigram-idf-stop | 0.284 | 0.537 | 0.397 | 0.659 |
| hybrid-bigram-tsrank | 0.373 | 0.761 | 0.536 | 0.920 |
| hybrid-bigram-idf | 0.701 | 0.910 | 0.791 | 0.963 |
| hybrid-bigram-idf-stop | 0.657 | 0.940 | 0.768 | 0.963 |
| hybrid-unigram-idf-stop | 0.642 | 0.881 | 0.748 | 0.948 |

### By question type: Hit@5 / MRR@10

| Configuration | lookup (n=15) | paraphrase (n=19) | numeric (n=23) | multi (n=10) |
|---|---|---|---|---|
| vector | 1.00 / 0.93 | 0.89 / 0.65 | 0.96 / 0.92 | 1.00 / 1.00 |
| keyword-bigram-tsrank | 0.13 / 0.05 | 0.26 / 0.14 | 0.35 / 0.13 | 0.60 / 0.52 |
| keyword-bigram-idf | 0.80 / 0.61 | 0.42 / 0.24 | 0.83 / 0.80 | 1.00 / 0.93 |
| keyword-bigram-idf-stop | 0.73 / 0.49 | 0.47 / 0.25 | 0.96 / 0.75 | 0.90 / 0.72 |
| keyword-unigram-idf-stop | 0.60 / 0.31 | 0.21 / 0.12 | 0.70 / 0.62 | 0.70 / 0.52 |
| hybrid-bigram-tsrank | 0.60 / 0.44 | 0.63 / 0.41 | 0.91 / 0.57 | 0.90 / 0.84 |
| hybrid-bigram-idf | 1.00 / 0.85 | 0.74 / 0.56 | 0.96 / 0.88 | 1.00 / 0.95 |
| hybrid-bigram-idf-stop | 1.00 / 0.78 | 0.79 / 0.51 | 1.00 / 0.92 | 1.00 / 0.88 |
| hybrid-unigram-idf-stop | 0.93 / 0.76 | 0.74 / 0.55 | 0.91 / 0.87 | 1.00 / 0.83 |

### Top-1 cosine similarity: answerable vs unanswerable

| Questions | n | min | median | max |
|---|---|---|---|---|
| answerable | 67 | 0.537 | 0.729 | 0.828 |
| unanswerable | 10 | 0.472 | 0.591 | 0.674 |

### Misses of the default configuration (vector, not in top 5)

- `q019` (numeric) Doanh nghiệp cho thuê lại lao động phải ký quỹ bao nhiêu tiền? Expected ND145/article/21; first relevant rank: 9; top 3: ND145/article/16, ND145/article/15, ND145/article/19
- `q040` (paraphrase) Chủ bắt nộp tiền cọc rồi mới cho vào làm thì có hợp pháp không? Expected BLLD2019/article/17; first relevant rank: not in top 10; top 3: BLLD2019/article/100, BLLD2019/article/94, ND145/article/19
- `q046` (paraphrase) Tôi nghỉ làm 5 ngày không xin phép thì có bị đuổi việc không? Expected BLLD2019/article/125; first relevant rank: not in top 10; top 3: ND145/article/65, BLLD2019/article/115, BLLD2019/article/113
<!-- retrieval:end -->
