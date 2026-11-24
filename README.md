# Vietnamese Labor Law Assistant (RAG)

Question answering over Vietnamese labor law (Bộ luật Lao động 2019, Nghị định 145/2020) with retrieval on PostgreSQL + pgvector, LangGraph orchestration and LLM answers that cite every article and clause.

> Work in progress: milestones M1 (corpus and parsing) and M2 (index and retrieval) are done. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and milestones.

**Not legal advice.** The corpus is the original text of Bộ luật Lao động số 45/2019/QH14 (passed 20/11/2019) and Nghị định số 145/2020/NĐ-CP (signed 14/12/2020), as published in the Official Gazette. Later amendments are **not** reflected yet (see [Future work](#future-work): one clause is already in force, a larger amending law takes effect on 01/03/2027).

## Corpus

| Document | Short id | Articles | Source |
|---|---|---|---|
| Bộ luật Lao động 45/2019/QH14 | `BLLD2019` | 220 (17 chapters) | [vanban.chinhphu.vn](https://vanban.chinhphu.vn/?pageid=27160&docid=198540), text from [Công báo 993+994](https://congbao.chinhphu.vn/van-ban/nghi-quyet-so-45-2019-qh14-30232.htm) |
| Nghị định 145/2020/NĐ-CP | `ND145` | 115 (+ 6 appendices) | [vanban.chinhphu.vn](https://vanban.chinhphu.vn/?pageid=27160&docid=201967), text from [Công báo 1203–1206](https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-145-2020-nd-cp-32732.htm) |

Vietnamese normative legal documents are not protected by copyright (Law on Intellectual Property, Art. 15), so the extracted text is kept in the repo.

## Quick start

Requires Python 3.10+ and Docker (for PostgreSQL + pgvector).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch first
pip install -e ".[dev]"            # app + tests
pip install -e ".[ingest]"         # only to re-extract text from the PDFs (PyMuPDF, AGPL)

# M1: corpus
python -m vnlaw_rag.cli fetch      # download the gazette PDFs to data/raw (3 requests)
python -m vnlaw_rag.cli parse      # PDF -> data/interim -> data/processed

# M2: index and retrieval
docker compose up -d db            # pgvector/pgvector:pg16 on 127.0.0.1:5442
python -m vnlaw_rag.cli index      # embed 348 chunks with bge-m3 and load them (one command)
python -m vnlaw_rag.cli search "Thời gian thử việc tối đa là bao lâu?"
python -m vnlaw_rag.cli eval       # retrieval metrics -> docs/EVAL.md

pytest                             # offline; DB integration tests skip if no database
ruff check . && ruff format --check .
```

- `parse` works without the PDFs or PyMuPDF: it then reuses the committed `data/interim/*.lines.jsonl`. It downloads the bge-m3 tokenizer file once (to count chunk tokens exactly).
- `index` downloads bge-m3 once (about 2.3 GB; on Windows without symlink support the Hugging Face cache keeps two weight files, about 4.5 GB). Embedding 348 chunks takes about 8 minutes on a laptop CPU; vectors are cached in `data/cache/`, so re-indexing is fast.
- Use `127.0.0.1` rather than `localhost` in `DATABASE_URL` on Windows: `localhost` tries IPv6 first and the container port is bound to IPv4 only, which costs a connection timeout.
- Configuration is read from environment variables or `.env` (see `.env.example`).

## Retrieval results

Draft golden set (77 questions, not yet reviewed by the owner, written by the coding agent: may be biased). Full tables, per-type numbers, date and git revision: [docs/EVAL.md](docs/EVAL.md).

| Configuration | Hit@5 | MRR@10 |
|---|---|---|
| **vector only, bge-m3 (default)** | 0.955 | 0.858 |
| keyword: syllable pairs + BM25-style IDF in SQL | 0.731 | 0.615 |
| keyword: syllable pairs + `ts_rank_cd` | 0.313 | 0.173 |
| hybrid RRF (vector + IDF keyword) | 0.940 | 0.768 |

Vector-only is the default because hybrid fusion did not beat it on this set ([decision 005](docs/decisions/005-retrieval-mode-vector-default.md)); hybrid stays available with `RETRIEVAL_MODE=hybrid`.

### Reviewing the golden set

```bash
python -m vnlaw_rag.cli review-export                      # data/eval/review-golden.xlsx, 25 questions
# fill the "Đánh giá" column (Đúng / Cần sửa / Bỏ), edit cells if needed, then:
python -m vnlaw_rag.cli review-import data/eval/review-golden.xlsx
pytest tests/test_eval.py                                  # re-checks evidence quotes
```

## Data layout

| Path | Content | Committed |
|---|---|---|
| `data/raw/*.pdf`, `data/raw/manifest.json` | Gazette PDFs; manifest with url, retrieved_at, sha256 | manifest only |
| `data/interim/<DOC>.lines.jsonl` | PDF text lines with bold flag and x position | yes |
| `data/processed/articles.jsonl` | One record per Điều, with nested Khoản / Điểm | yes |
| `data/processed/appendices.jsonl` | Phụ lục (lists in full, forms as a table of contents) | yes |
| `data/processed/chunks.jsonl` | Retrieval chunks with metadata | yes |
| `data/processed/build_info.json` | Counts, parameters, input hashes, parser anomalies | yes |
| `data/cache/*.npy` | Cached chunk embeddings | no |
| `data/eval/golden.jsonl` | Golden set: questions, expected articles, evidence quotes, reference answers | yes |
| `data/eval/retrieval_results.json` | Per-question ranks of the last retrieval eval | yes |
| `data/eval/review-golden.xlsx` | Review workbook for the owner | yes |

## Design decisions

- [001 Corpus source and PDF extraction](docs/decisions/001-corpus-source-and-pdf-extraction.md)
- [002 Layout-aware legal parser](docs/decisions/002-layout-aware-legal-parser.md)
- [003 Article-level chunking](docs/decisions/003-article-level-chunking.md)
- [004 Own pgvector SQL instead of langchain-postgres](docs/decisions/004-own-pgvector-sql-instead-of-langchain-postgres.md)
- [005 Default retrieval: vector-only, hybrid measured](docs/decisions/005-retrieval-mode-vector-default.md)

## Future work

Known amendments not yet in the corpus (checked 2026-11-24 on official sources):

- **Bộ luật Lao động, in force since 01/01/2026:** clause 8a added to Điều 154 (high-quality digital technology workforce exempt from work permits) by khoản 4 Điều 49 of Luật Công nghiệp công nghệ số 71/2025/QH15. It is the only change in the consolidated text 125/VBHN-VPQH (27/08/2025, [Công báo 1309+1310](https://congbao.chinhphu.vn/van-ban/van-ban-hop-nhat-so-125-vbhn-vpqh-46084.htm)); an article-by-article comparison found no other substantive difference.
- **Bộ luật Lao động, from 01/03/2027:** the law amending the Labor Code and four other laws, passed on 21/11/2026 ([Báo Chính phủ](https://baochinhphu.vn/quoc-hoi-dong-y-bo-sung-2-ngay-nghi-le-duoc-huong-nguyen-luong-102261121121448218.htm)): labor leasing moves from a license to deposit + notification, two new paid public holidays (27/7 and 24/11), and changes to collective bargaining, the National Wage Council, labor management and strikes. Article list to be added when the text is published in Công báo.
- **Nghị định 145/2020:** a draft amending decree was published for comment in July 2026 (labor leasing, Phụ lục II); not issued as far as we found.

## License

MIT for the code. PyMuPDF (AGPL-3.0) is an optional, offline ingest dependency and is not used by the application.
