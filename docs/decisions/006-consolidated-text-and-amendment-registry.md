# 006. Labor Code from its consolidated text, plus an amendment registry with effective dates

Date: 2026-11-24 · Status: accepted (coordinator decision) · Updates 001

## Context
The corpus held the Labor Code as promulgated in 2019. An assistant that answers from a superseded clause is wrong in a way that is easy to catch in a demo. Two facts, checked on official sources: (1) the consolidated text 125/VBHN-VPQH (27/08/2025, Công báo 1309+1310, clean text layer) includes the only amendment so far, clause 8a of Điều 154, added by Luật 71/2025/QH15 and in force since 01/01/2026; an article-by-article comparison found no other substantive difference. (2) On 21/11/2026 the National Assembly passed a law amending the Labor Code (labor leasing, public holidays, collective bargaining, the National Wage Council, strikes), effective 01/03/2027; its text is not yet in Công báo.

## Decision
- The Labor Code text comes from VBHN 125, the law in force today. The parser now handles what consolidated texts add: superscript footnote markers (removed from the text, recorded), footnote lines (12 pt, kept out of the body, attached as notes to the clause they annotate) and inserted clauses ("8a", checked like any other number).
- `data/amendments.json` lists every known amending document: status `in_force | passed_pending_text | draft`, effective date, affected articles, official source URL and access date. Articles are `explicit` when taken from a VBHN footnote and `inferred_from_topic` when the official source only names the topic. Drafts are recorded but never used in answers.
- At build time each chunk gets `amended_by` (changes already in its text) and `upcoming_change` (announced changes with their effective date, hedged when inferred). M3/M4 must surface them when the chunk is cited (done: `warnings` in the API and the UI, decision 007); the notice reads "Theo VBHN 125/VBHN-VPQH (2025); một số quy định sẽ thay đổi từ 01/03/2027."

## Alternatives
- Keep the 2019 text and only flag Điều 154: cheaper, but answers would miss a clause in force.
- Wait for a new VBHN: consolidated texts lag months behind; until then answers would be silently out of date.

## Consequences
- Retrieval metrics did not change: only 5 chunk texts changed (Điều 42, 125, 154, 219, 220). With the per-text embedding cache, re-indexing took 9.6 s instead of about 8 minutes.
- Known quirks of VBHN 125, kept as printed: the heading of Điều 64 is not bold (accepted and logged as the one parser anomaly), Điều 42 khoản 2 lacks the a)/b) labels, a stray "X" in Điều 219, "ma tuý" spelling, "/." ending.
- When the amending law's text is published: add its number and exact articles to the registry (mapping becomes explicit), and later switch to the next VBHN. Golden items carry `as_of` / `valid_until` for answers that will expire.
