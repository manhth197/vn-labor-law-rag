# 001. Corpus source: Official Gazette PDFs, text extracted with PyMuPDF

Date: 2026-11-24 · Status: accepted

## Context
We need the exact text of Bộ luật Lao động 2019 (45/2019/QH14) and Nghị định 145/2020/NĐ-CP from an official portal. vbpl.vn was rebuilt as a JavaScript app behind an anti-bot layer and its old full-text URLs now redirect to the home page. vanban.chinhphu.vn links signed PDFs, but they are image scans with no text layer (OCR would add errors to legal text). The Official Gazette (Công báo, congbao.chinhphu.vn) publishes text-layer PDFs; texts published in Công báo are official and carry the same value as the original. The decree spans two gazette issues (1203+1204 and 1205+1206).

## Decision
- Download the three Công báo PDFs (`python -m vnlaw_rag.cli fetch`): generic User-Agent, 1 s between requests, sha256 pinned in `ingest/sources.py` and recorded in `data/raw/manifest.json`. PDFs are not committed (gitignored); the fetch command recreates them.
- Extract lines with **PyMuPDF**, keeping a bold flag and the x position of each line (used by the parser, see 002). Measured on the Labor Code: PyMuPDF finds 220/220 "Điều" headings with clean syllables in 0.1 s; pypdf finds 220 but inserts spaces inside syllables ("vi ệc"); pdfminer.six finds only 207 and takes ~10 s.
- PyMuPDF is AGPL-3.0 and the repo is MIT. It is used only offline at ingest time: it sits in the optional extra `[ingest]`, is imported lazily in one module (`ingest/pdf_text.py`), and a test checks that the pipeline, parser and CLI load without it. Its output (`data/interim/*.lines.jsonl`) is committed, so parsing, tests, CI and the app never need PyMuPDF or the PDFs.

## Alternatives
- OCR of the signed scans: extra system dependency (Tesseract) and character errors in diacritics.
- Third-party sites with clean HTML (e.g. thuvienphapluat.vn): terms of use restrict scraping; not an official source.
- The gazette `.doc` files: legacy binary Word format, needs LibreOffice or antiword to convert.

## Consequences
- The corpus is the **original text as promulgated**, not a consolidated version: later amendments are not reflected. Stated in README, ARCHITECTURE and (M4) the UI disclaimer. Future work: switch the Labor Code to the consolidated text 125/VBHN-VPQH (27/08/2025, Công báo 1309+1310); check whether Nghị định 145 has been amended (a draft amending decree was published for comment in July 2026).
- If an upstream PDF changes, `fetch` reports "CHANGED UPSTREAM" and the parser tests catch structural drift.
