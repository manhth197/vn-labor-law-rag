"""The amendment registry (data/amendments.json) and the chunk annotations built from it."""

from __future__ import annotations

import json
from urllib.parse import urlparse

import pytest

from tests.conftest import REPO_ROOT, parsed, word_count
from vnlaw_rag.ingest.amendments import annotate_chunks, corpus_notice, load_registry
from vnlaw_rag.ingest.chunk import build_chunks

REGISTRY = load_registry(REPO_ROOT / "data" / "amendments.json")
OFFICIAL = ("quochoi.vn", "chinhphu.vn", "baochinhphu.vn", "vbpl.vn")


def _official(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return any(host == d or host.endswith("." + d) for d in OFFICIAL)


def test_every_entry_has_an_official_source_with_access_date():
    for a in REGISTRY.amendments:
        assert a.sources, a.id
        for source in a.sources:
            assert _official(source.url), (a.id, source.url)
            assert source.accessed


def test_statuses_and_answer_use_are_consistent():
    by_id = {a.id: a for a in REGISTRY.amendments}
    assert by_id["luat-71-2025-qh15"].status == "in_force"
    assert by_id["luat-71-2025-qh15"].in_corpus_text
    pending = by_id["luat-sua-doi-blld-2026"]
    assert (pending.status, pending.effective_date) == ("passed_pending_text", "2027-03-01")
    assert not pending.in_corpus_text
    for a in REGISTRY.amendments:
        if a.status == "draft":
            assert not a.use_in_answers and not a.affects, a.id


def test_affected_articles_exist_in_the_corpus():
    articles = {
        (a.doc_short, a.article_no) for d in ("BLLD2019", "ND145") for a in parsed(d).articles
    }
    for a in REGISTRY.amendments:
        for effect in a.affects:
            for n in effect.articles:
                assert (effect.doc, n) in articles, (a.id, effect.doc, n)


def test_explicit_entries_match_the_footnotes_of_the_consolidated_text():
    blld = {a.article_no: a for a in parsed("BLLD2019").articles}
    for a in REGISTRY.amendments:
        for effect in a.affects:
            if effect.mapping != "explicit" or not a.in_corpus_text:
                continue
            for n in effect.articles:
                notes = blld[n].notes
                assert {note.clause for note in notes} >= set(effect.clauses)
                assert all(a.amending.number in note.text for note in notes)


@pytest.fixture(scope="module")
def chunks():
    docs = [parsed("BLLD2019"), parsed("ND145")]
    built = build_chunks(
        [a for d in docs for a in d.articles],
        [a for d in docs for a in d.appendices],
        word_count,
        max_tokens=1200,
    )
    annotate_chunks(built, REGISTRY)
    return {c.chunk_id: c for c in built}


def test_amended_article_is_flagged_with_its_effective_date(chunks):
    (message,) = chunks["BLLD2019-D154"].amended_by
    assert "Khoản 8a" in message and "71/2025/QH15" in message and "01/01/2026" in message
    assert chunks["BLLD2019-D154"].upcoming_change == []


def test_upcoming_change_is_flagged_and_hedged(chunks):
    (message,) = chunks["BLLD2019-D112"].upcoming_change
    assert message.startswith("Sẽ thay đổi từ 01/03/2027:")
    assert "27/7" in message and "24/11" in message
    assert "có thể bị ảnh hưởng; chưa có toàn văn" in message
    assert chunks["ND145-D21"].upcoming_change  # labor-leasing license
    assert chunks["BLLD2019-D25"].upcoming_change == []
    assert chunks["BLLD2019-D25"].amended_by == []


def test_draft_decree_never_reaches_chunks(chunks):
    texts = json.dumps([c.model_dump() for c in chunks.values()], ensure_ascii=False)
    assert "Dự thảo" not in texts


def test_corpus_notice():
    notice = corpus_notice(REGISTRY, ["VBHN 125/VBHN-VPQH (2025)"])
    assert notice == "Theo VBHN 125/VBHN-VPQH (2025); một số quy định sẽ thay đổi từ 01/03/2027."
