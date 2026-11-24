# 009. Answer evaluation: deterministic metrics first, LLM judge second

Date: 2026-11-24 · Status: accepted (coordinator) · Run: git 830902c, `eval-answers`

## Context
Retrieval scores (Hit@k, MRR) say nothing about the final answer. The free-tier Gemini quota is small and noisy, the golden set was drafted by the coding agent, and the only affordable judge is the generator model itself, which may favor its own answers.

## Decision
- **Deterministic metrics are the headline:** article-level citation precision and recall against the golden `expected` refs; refusal correctness (answerable questions must be answered, out-of-scope ones refused); numeric exact match of one key figure per numeric question after normalization (leading zeros, `2 tỷ` = `2.000.000.000`, number words, `,` decimals).
- **LLM judge is secondary:** reference-based, 1-5 for correctness/completeness and faithfulness to the cited sources, only on answered questions, reported with the caveat "same model as the generator".
- **Reported twice:** on the 22 owner-reviewed answerable questions and on all 77, with date, git revision, model, request count and error rate.
- **Cheap and safe to run:** one model, no fallback, sequential with 4 s between requests, 180 s budget per answer; stop if more than 20% of the first 20 requests fail or on a rate limit; results are appended to a JSONL file so a run resumes where it stopped. `--report-only` re-scores stored answers without any API call.

## Result (148 requests, 11 failed = 7.4%, 77/77 scored)
Reviewed: answered 0.864, citation precision 0.974 / recall 0.947, numeric 7/8, judge 5.00 / 4.95. All: answered 0.940, precision 0.947 / recall 0.976, numeric 22/23, out-of-scope refused 10/10, judge 4.94 / 4.97. All 4 wrong refusals are retrieval misses, not generation errors: q019 (right article at rank 9, only 6 are sent), q040 and q046 (not in the top 10), q052 (refused by `grade`, top-1 cosine 0.537). In the 3 cases that reached it, the model refused instead of inventing an answer.

## Changes made after the run (recorded in `answer_run.json`)
- `GRADE_MIN_SCORE` 0.55 -> 0.50: 0.55 sat above the lowest answerable top-1 score (0.537). At 0.50, `grade` still stops 1 of the 10 out-of-scope questions before the LLM (was 2); the others rely on the model's `sufficient` flag, which refused 8/8 in this run. Only the questions with a top-1 cosine in [0.50, 0.55) can change, so only q052 and q073 were re-run (git aed54ed, 3 requests, 0 failed): q052 is now answered correctly, q073 is refused by the model. Current configuration: reviewed answered 0.909 (was 0.864), all 0.955 (was 0.940); the full run stays in EVAL.md as history.
- q011 got `key_fact: "48 giờ"`: the question asks per week, the first figure of the reference answer is per day. Numeric match was re-scored from the stored answers (21 -> 22 of 23).
- The recorded revision was corrected from `830902c-dirty` to `830902c`: the working tree was edited while the run was in progress; the revision is now read at start.

## Consequences
- The next gain is retrieval for paraphrased questions (reranker or query rewriting), not prompting.
- A judge from another provider, or a human check of a sample, would make the 1-5 scores meaningful; until then they are a sanity check only.
