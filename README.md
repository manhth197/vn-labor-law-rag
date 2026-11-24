# Vietnamese Labor Law Assistant (RAG)

Question answering over Vietnamese labor law (Bộ luật Lao động 2019, Nghị định 145/2020) with hybrid retrieval on pgvector, LangGraph orchestration and Claude answers that cite every article and clause.

> Work in progress: milestone M1 (corpus and parsing) is done. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and milestones.

**Not legal advice.** The corpus is the original text of Bộ luật Lao động số 45/2019/QH14 (passed 20/11/2019) and Nghị định số 145/2020/NĐ-CP (signed 14/12/2020), as published in the Official Gazette. Later amendments are **not** reflected yet (see [Future work](#future-work)).

## Corpus

| Document | Short id | Articles | Source |
|---|---|---|---|
| Bộ luật Lao động 45/2019/QH14 | `BLLD2019` | 220 (17 chapters) | [vanban.chinhphu.vn](https://vanban.chinhphu.vn/?pageid=27160&docid=198540), text from [Công báo 993+994](https://congbao.chinhphu.vn/van-ban/nghi-quyet-so-45-2019-qh14-30232.htm) |
| Nghị định 145/2020/NĐ-CP | `ND145` | 115 (+ 6 appendices) | [vanban.chinhphu.vn](https://vanban.chinhphu.vn/?pageid=27160&docid=201967), text from [Công báo 1203–1206](https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-145-2020-nd-cp-32732.htm) |

Vietnamese normative legal documents are not protected by copyright (Law on Intellectual Property, Art. 15), so the extracted text is kept in the repo.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"            # app + tests
pip install -e ".[ingest]"         # only to re-extract text from the PDFs (PyMuPDF, AGPL)

python -m vnlaw_rag.cli fetch      # download the gazette PDFs to data/raw (3 requests)
python -m vnlaw_rag.cli parse      # PDF -> data/interim -> data/processed
pytest                             # tests run offline on the committed data
ruff check . && ruff format --check .
```

`parse` works without the PDFs or PyMuPDF: it then reuses the committed `data/interim/*.lines.jsonl`. It downloads the bge-m3 tokenizer file once (to count chunk tokens exactly).

## Data layout

| Path | Content | Committed |
|---|---|---|
| `data/raw/*.pdf`, `data/raw/manifest.json` | Gazette PDFs; manifest with url, retrieved_at, sha256 | manifest only |
| `data/interim/<DOC>.lines.jsonl` | PDF text lines with bold flag and x position | yes |
| `data/processed/articles.jsonl` | One record per Điều, with nested Khoản / Điểm | yes |
| `data/processed/appendices.jsonl` | Phụ lục (lists in full, forms as a table of contents) | yes |
| `data/processed/chunks.jsonl` | Retrieval chunks with metadata | yes |
| `data/processed/build_info.json` | Counts, parameters, input hashes, parser anomalies | yes |

## Design decisions

- [001 Corpus source and PDF extraction](docs/decisions/001-corpus-source-and-pdf-extraction.md)
- [002 Layout-aware legal parser](docs/decisions/002-layout-aware-legal-parser.md)
- [003 Article-level chunking](docs/decisions/003-article-level-chunking.md)

## Future work

- Use consolidated texts so amendments are reflected: the Labor Code has a consolidated text, 125/VBHN-VPQH (27/08/2025, [Công báo 1309+1310](https://congbao.chinhphu.vn/van-ban/van-ban-hop-nhat-so-125-vbhn-vpqh-46084.htm)). For Nghị định 145, a draft amending decree was published for comment in July 2026; check whether it has been issued.

## License

MIT for the code. PyMuPDF (AGPL-3.0) is an optional, offline ingest dependency and is not used by the application.
