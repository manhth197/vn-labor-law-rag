# Evaluation

No invented numbers: every table below is produced by a command and records its date, git revision and configuration.

<!-- retrieval:start -->
## Retrieval

Run: 2026-11-24 · git `34f32e8` · command `python -m vnlaw_rag.cli eval`

- Embeddings: `BAAI/bge-m3` (normalized, cosine, HNSW). Chunks: 348 (max 1200 tokens), chunks.jsonl sha256 `bde32be6ba62`.
- Candidates: vector top 20, keyword top 20; RRF k = 60. Metrics at depth 10, article-level matching.
- Golden set: `data/eval/golden.jsonl`, 77 questions (numeric 23, lookup 15, paraphrase 19, multi 10, unanswerable 10). **Reviewed by the owner: 22/77.** Unanswerable questions are excluded from the ranking metrics.

> **Caveat (golden set do agent soạn, có thể thiên lệch).** The questions were drafted by the coding agent from the article texts, not by users. Lookup items reuse the law's wording, and there are few keyword-heavy questions (exact terms, document numbers). The owner reviewed a stratified sample of 25 and accepted 22; the other questions are still unreviewed.

### Reviewed subset (owner-reviewed answerable questions)

Small sample: n = 22, so one question moves Hit@k by 4.5 points. No unanswerable question is in the reviewed subset.

| Configuration | n | Hit@1 | Hit@5 | MRR@10 | Recall@10 |
|---|---|---|---|---|---|
| **vector** (default) | 22 | 0.864 | 0.909 | 0.891 | 0.955 |
| keyword-bigram-tsrank | 22 | 0.045 | 0.227 | 0.152 | 0.417 |
| keyword-bigram-idf | 22 | 0.591 | 0.773 | 0.667 | 0.818 |
| keyword-bigram-idf-stop | 22 | 0.409 | 0.864 | 0.587 | 0.848 |
| keyword-unigram-idf-stop | 22 | 0.227 | 0.500 | 0.345 | 0.614 |
| hybrid-bigram-tsrank | 22 | 0.455 | 0.818 | 0.608 | 0.894 |
| hybrid-bigram-idf | 22 | 0.773 | 0.909 | 0.827 | 0.955 |
| hybrid-bigram-idf-stop | 22 | 0.773 | 0.955 | 0.829 | 0.955 |
| hybrid-unigram-idf-stop | 22 | 0.545 | 0.864 | 0.667 | 0.909 |

Configuration names: `keyword-bigram-tsrank` = OR of syllable pairs ranked by `ts_rank_cd`; `-idf` = BM25-style IDF weighting computed in SQL; `-stop` = terms with Vietnamese function words dropped; `hybrid-*` = that keyword leg fused with the vector leg by RRF.

### All answerable questions (reviewed + unreviewed)

| Configuration | n | Hit@1 | Hit@5 | MRR@10 | Recall@10 |
|---|---|---|---|---|---|
| **vector** (default) | 67 | 0.776 | 0.955 | 0.858 | 0.970 |
| keyword-bigram-tsrank | 67 | 0.075 | 0.313 | 0.173 | 0.455 |
| keyword-bigram-idf | 67 | 0.537 | 0.731 | 0.615 | 0.806 |
| keyword-bigram-idf-stop | 67 | 0.388 | 0.761 | 0.545 | 0.811 |
| keyword-unigram-idf-stop | 67 | 0.284 | 0.537 | 0.397 | 0.659 |
| hybrid-bigram-tsrank | 67 | 0.373 | 0.761 | 0.536 | 0.920 |
| hybrid-bigram-idf | 67 | 0.701 | 0.910 | 0.791 | 0.963 |
| hybrid-bigram-idf-stop | 67 | 0.657 | 0.940 | 0.768 | 0.963 |
| hybrid-unigram-idf-stop | 67 | 0.642 | 0.881 | 0.748 | 0.948 |

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

<!-- answers-current:start -->
<!-- answers-current:end -->

<!-- answers:start -->
## Answers: full run

Run: 2026-11-24 · git `830902c` · command `python -m vnlaw_rag.cli eval-answers`

- Generator: `gemini-3.5-flash-lite` (main model only, no fallback, 180 s budget); judge: `gemini-3.5-flash-lite`. Sequential, 4 s between API requests.
- API requests: 148 (generator + judge), failed: 11 (7.4%). Questions scored: 77 of 77.
- Note: git: the process ran commit 830902c; the end-of-run value '830902c-dirty' came from edits made to the working tree while it ran (fixed: the revision is now taken at start).
- Note: q011: key figure annotated as '48 giờ' after the run (the question asks per week; the first-figure rule had picked '8 giờ'); numeric rows re-scored from the stored answers, no API call.
- Note: After this run GRADE_MIN_SCORE was lowered from 0.55 to 0.50: q052 (answerable, top-1 cosine 0.537, right article at rank 2) was refused by `grade`. Not re-run.

> **Caveats.** The judge is the same model as the generator, so it may favor its own answers: the deterministic rows are the main numbers. The golden set was drafted by the coding agent; the owner reviewed 22 answerable questions. Numeric match checks one key figure per question: the item's `key_fact` if annotated, else the first figure of the reference answer.

| Metric | Reviewed (n=22) | All (n=77) |
|---|---|---|
| Answerable questions scored | 22 | 67 |
| Answered (not refused) | 0.864 | 0.940 |
| False refusals | 0.136 | 0.060 |
| Citation precision (article level) | 0.974 | 0.947 |
| Citation recall (article level) | 0.947 | 0.976 |
| Cites at least one expected article | 1.000 | 1.000 |
| Numeric questions | 8 | 23 |
| Numeric exact match (key figure) | 0.875 | 0.957 |
| Unanswerable questions | 0 | 10 |
| Unanswerable correctly refused | - | 1.000 |
| Judged answers | 19 | 63 |
| Judge: correctness 1-5 (secondary) | 5.00 | 4.94 |
| Judge: faithfulness 1-5 (secondary) | 4.95 | 4.97 |

### Errors, wrong refusals and numeric misses

- `q019` (numeric): wrong refusal; key figure '2000000000 đồng' missing
- `q040` (paraphrase): wrong refusal
- `q046` (paraphrase): wrong refusal
- `q052` (paraphrase): wrong refusal

Per-question answers and scores: `data/eval/answer_results.jsonl`.
<!-- answers:end -->

### Answer misses: manual analysis (2026-11-24)

Written by hand from `data/eval/answer_results.jsonl` and the retrieval results above; not generated.

- All 4 wrong refusals come from retrieval, and the model refused rather than invent an answer: `q019` (right article at rank 9; only the top 6 chunks go to the model), `q040` and `q046` (paraphrased questions whose article is not in the top 10), `q052` (top-1 cosine 0.537 under the `grade` threshold 0.55, right article at rank 2).
- After the run the threshold was lowered to 0.50 (decision 009), so `q052` would now reach the model; not re-run.
- `q011` was a scoring artifact: the reference starts with the per-day limit (8 giờ) while the question asks per week (48 giờ, which the answer gave). It now carries `key_fact: "48 giờ"`.
