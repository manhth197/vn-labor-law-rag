# 002. Layout-aware parser for Chương / Mục / Điều / Khoản / Điểm

Date: 2026-11-24 · Status: accepted

## Context
Citations must point to the exact Điều and Khoản, so the parser has to recover the legal structure from PDF lines without mistakes. Plain-text regexes are fragile: a wrapped line can start with "2." ("... theo khoản\n2. Điều 5"), titles wrap over two lines, and amending articles quote other laws ("“Điều 54. ...”") which look like headings.

## Decision
A line-based state machine that uses four signals measured on the Công báo layout:
1. **Bold** = heading. "Chương", "Mục", "Điều N." and their wrapped titles are bold; body text is not.
2. **Indentation** = new paragraph. Body left margin is 64 pt, first-line indent 87 pt. A clause/point marker counts only on an indented line.
3. **Sequential numbering.** A clause is accepted only if it is the next number, a point only if it is the next letter of the Vietnamese sequence (a, b, c, d, đ, e, g, ...). Articles must be strictly sequential or parsing fails loudly.
4. **Quotes.** Text inside “...” is never structure.
Anything that looks like structure but breaks a rule is kept as text and logged as an **anomaly** in `build_info.json` (0 on the original texts; 1 accepted on the consolidated text, see 006). Page headers, page numbers and the signature stamp are dropped by position (top band y < 75 pt). Front matter is skipped up to the first heading of each gazette issue; the body ends at the signature block. Appendices: substantive lists (Phụ lục II, VI of the decree) are kept in full; form collections keep only their table of contents.

## Alternatives
- Regex over plain text only: no bold/indent signal, so wrapped titles and wrapped "N." lines need guesswork.
- LLM-based structuring: costs money, is not deterministic and is hard to test.

## Consequences
- Tested on the real text: 220 articles / 17 chapters (Labor Code), 115 articles (decree), every clause and point sequence valid.
- The decree prints "Chương XI" twice (chapter 11 and the final chapter). Verified against the signed scan (page 86): the error is in the original, not the gazette. We keep the printed number and add `chapter_seq` (1..12) for ordering; a repeated heading with the same title at the start of the second gazette issue is treated as a continuation.
- The thresholds (75 pt band, 64/87 pt margins) are specific to the Công báo layout; a new source needs re-measuring.
