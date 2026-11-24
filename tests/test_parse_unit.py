"""Parser edge cases on small synthetic inputs."""

from __future__ import annotations

import dataclasses

import pytest

from tests.conftest import body, bold, para
from vnlaw_rag.ingest.models import Line
from vnlaw_rag.ingest.parse import ParseError, parse_document
from vnlaw_rag.ingest.sources import BLLD2019

DOC = dataclasses.replace(BLLD2019, body_end_patterns=(r"^KẾT THÚC$",))


def run(lines):
    """Parse `lines` followed by a signature line in the last source part."""
    last_part = max(line.part for line in lines)
    return parse_document([*lines, bold("KẾT THÚC", part=last_part)], DOC)


def test_front_matter_is_skipped_and_headings_are_parsed():
    result = run(
        [
            bold("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM"),
            para("Căn cứ Hiến pháp;"),
            bold("Chương I", x0=279),
            bold("NHỮNG QUY ĐỊNH", x0=250),
            bold("CHUNG", x0=270),
            bold("Mục 1", x0=280),
            bold("GIAO KẾT", x0=260),
            bold("Điều 1. Phạm vi điều chỉnh của"),
            bold("bộ luật", x0=64),
            para("Bộ luật này quy định"),
            body("tiêu chuẩn lao động."),
        ]
    )
    (a,) = result.articles
    assert (a.chapter_no, a.chapter_title) == ("I", "NHỮNG QUY ĐỊNH CHUNG")
    assert (a.section_no, a.section_title) == (1, "GIAO KẾT")
    assert a.article_title == "Phạm vi điều chỉnh của bộ luật"
    assert a.intro == "Bộ luật này quy định tiêu chuẩn lao động."
    assert result.anomalies == []


def test_clause_number_at_start_of_wrapped_line_is_text():
    result = run(
        [
            bold("Chương I"),
            bold("A"),
            bold("Điều 1. Tiêu đề"),
            para("1. Theo quy định tại khoản"),
            body("2. Điều 5 của Bộ luật này."),  # not indented: a wrapped line, not clause 2
            para("2. Khoản thứ hai."),
        ]
    )
    clauses = result.articles[0].clauses
    assert [c.no for c in clauses] == [1, 2]
    assert clauses[0].text == "Theo quy định tại khoản 2. Điều 5 của Bộ luật này."


def test_out_of_sequence_clause_is_kept_as_text_and_reported():
    result = run(
        [
            bold("Chương I"),
            bold("A"),
            bold("Điều 1. Tiêu đề"),
            para("1. Một."),
            para("3. Ba."),
        ]
    )
    a = result.articles[0]
    assert [c.no for c in a.clauses] == [1]
    assert a.clauses[0].text == "Một.\n3. Ba."
    (anomaly,) = result.anomalies
    assert "expected 2" in anomaly.message


def test_points_follow_vietnamese_alphabet():
    letters = ["a", "b", "c", "d", "đ", "e", "g"]
    result = run(
        [
            bold("Chương I"),
            bold("A"),
            bold("Điều 1. Tiêu đề"),
            para("1. Gồm:"),
            *[para(f"{x}) Mục {x};") for x in letters],
        ]
    )
    assert [p.letter for p in result.articles[0].clauses[0].points] == letters


def test_points_without_a_clause_attach_to_the_article():
    result = run(
        [bold("Chương I"), bold("A"), bold("Điều 1. T"), para("Gồm:"), para("a) X;"), para("b) Y.")]
    )
    a = result.articles[0]
    assert a.intro == "Gồm:"
    assert [p.letter for p in a.intro_points] == ["a", "b"]


def test_quoted_text_is_never_structure():
    result = run(
        [
            bold("Chương I"),
            bold("A"),
            bold("Điều 1. Sửa đổi luật khác"),
            para("1. Sửa đổi Điều 54 như sau:"),
            bold("“Điều 54. Điều kiện"),
            para("1. Nội dung được trích dẫn;"),
            para("a) điểm trích dẫn.”;"),
            para("2. Khoản thật thứ hai."),
        ]
    )
    a = result.articles[0]
    assert [c.no for c in a.clauses] == [1, 2]
    assert "“Điều 54. Điều kiện\n1. Nội dung được trích dẫn;\na) điểm trích dẫn.”;" in (
        a.clauses[0].text
    )
    assert result.anomalies == []


def test_repeated_chapter_heading_is_a_continuation_but_new_title_is_a_new_chapter():
    result = run(
        [
            bold("Chương XI"),
            bold("TRANH CHẤP"),
            bold("Điều 1. A"),
            para("Nội dung."),
            bold("Chương XI", part=1),  # next gazette issue repeats the heading
            bold("TRANH CHẤP", part=1),
            bold("Điều 2. B", part=1),
            para("Nội dung.", part=1),
            bold("Chương XI", part=1),  # printed number repeats, title differs
            bold("ĐIỀU KHOẢN THI HÀNH", part=1),
            bold("Điều 3. C", part=1),
            para("Nội dung.", part=1),
        ]
    )
    seqs = [(a.chapter_seq, a.chapter_no, a.chapter_title) for a in result.articles]
    assert seqs == [
        (1, "XI", "TRANH CHẤP"),
        (1, "XI", "TRANH CHẤP"),
        (2, "XI", "ĐIỀU KHOẢN THI HÀNH"),
    ]


def test_continuation_part_front_matter_is_skipped():
    result = run(
        [
            bold("Chương I"),
            bold("A"),
            bold("Điều 1. A"),
            para("Nội dung."),
            bold("VĂN BẢN QUY PHẠM PHÁP LUẬT", part=1),
            para("(Tiếp theo Công báo số 1203 + 1204)", part=1),
            bold("Điều 2. B", part=1),
            para("Nội dung hai.", part=1),
        ]
    )
    assert [a.article_no for a in result.articles] == [1, 2]
    assert result.articles[0].intro == "Nội dung."


def test_article_numbers_must_be_sequential():
    with pytest.raises(ParseError, match="expected Điều 2"):
        run([bold("Chương I"), bold("A"), bold("Điều 1. A"), bold("Điều 3. C")])


def test_missing_signature_block_is_an_error():
    with pytest.raises(ParseError, match="end of body"):
        parse_document([bold("Chương I"), bold("A"), bold("Điều 1. A")], DOC)


def test_inserted_clauses_follow_their_base_number():
    result = run(
        [
            bold("Chương I"),
            bold("A"),
            bold("Điều 1. T"),
            para("1. Một."),
            para("1a. Chèn thêm."),
            para("1b. Chèn tiếp."),
            para("2. Hai."),
            para("2b. Sai thứ tự."),
        ]
    )
    a = result.articles[0]
    assert [c.label for c in a.clauses] == ["1", "1a", "1b", "2"]
    (anomaly,) = result.anomalies
    assert "looks like clause 2b" in anomaly.message


def test_footnotes_leave_the_body_and_attach_to_their_clause():
    def note(text, *, markers=()):
        return Line(
            page=1, bold=False, x0=87, text=text, part=0, footnote=True, markers=list(markers)
        )

    result = run(
        [
            bold("Chương I"),
            bold("A"),
            bold("Điều 1. T"),
            para("1. Một."),
            Line(page=1, bold=False, x0=87, text="1a. Chèn thêm.", part=0, markers=["2"]),
            note("Khoản này được bổ sung theo", markers=["2"]),
            note("Luật số 1/2025/QH15."),
            para("2. Hai."),
        ]
    )
    a = result.articles[0]
    assert [c.label for c in a.clauses] == ["1", "1a", "2"]
    assert a.clauses[1].text == "Chèn thêm."
    assert [(n.clause, n.text) for n in a.notes] == [
        ("1a", "Khoản này được bổ sung theo Luật số 1/2025/QH15.")
    ]
    assert result.footnotes == {"2": "Khoản này được bổ sung theo Luật số 1/2025/QH15."}
    assert "bổ sung" not in a.body_text()


def test_unbolded_heading_accepted_only_as_the_next_indented_article():
    result = run(
        [
            bold("Chương I"),
            bold("A"),
            bold("Điều 1. Một"),
            para("Theo quy định tại"),
            body("Điều 2. của luật khác."),  # not indented: a wrapped line, stays text
            para("Điều 3. Không phải Điều kế tiếp."),  # wrong number: text
            para("Điều 2. Hai"),  # next number, indented, regular weight: heading
            para("Nội dung hai."),
        ]
    )
    assert [a.article_no for a in result.articles] == [1, 2]
    assert result.articles[1].article_title == "Hai"
    assert "Điều 3. Không phải Điều kế tiếp." in result.articles[0].intro
    assert [x.message for x in result.anomalies] == [
        "article heading not bold in the source; accepted"
    ]
