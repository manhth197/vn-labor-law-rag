"""Parser checks on the real documents (committed data/interim lines)."""

from __future__ import annotations

from vnlaw_rag.ingest.parse import POINT_LETTERS

ROMAN = [
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII",
]  # fmt: skip


def test_labor_code_has_220_sequential_articles(blld):
    assert [a.article_no for a in blld.articles] == list(range(1, 221))


def test_labor_code_has_17_chapters_in_order(blld):
    chapters = []
    for a in blld.articles:
        if not chapters or chapters[-1] != (a.chapter_seq, a.chapter_no):
            chapters.append((a.chapter_seq, a.chapter_no))
    assert chapters == list(enumerate(ROMAN, start=1))


def test_labor_code_article_titles(blld):
    titles = {a.article_no: a.article_title for a in blld.articles}
    assert titles[1] == "Phạm vi điều chỉnh"
    assert titles[25] == "Thời gian thử việc"
    assert titles[35] == "Quyền đơn phương chấm dứt hợp đồng lao động của người lao động"
    assert titles[98] == "Tiền lương làm thêm giờ, làm việc vào ban đêm"
    assert titles[99] == "Tiền lương ngừng việc"
    assert titles[113] == "Nghỉ hằng năm"
    assert titles[169] == "Tuổi nghỉ hưu"
    assert titles[220] == "Hiệu lực thi hành"
    # A title that wraps over two lines in the PDF is joined back together.
    assert titles[43] == (
        "Nghĩa vụ của người sử dụng lao động khi chia, tách, hợp nhất, sáp nhập; bán, cho "
        "thuê, chuyển đổi loại hình doanh nghiệp; chuyển nhượng quyền sở hữu, quyền sử dụng "
        "tài sản của doanh nghiệp, hợp tác xã"
    )


def test_titles_are_clean(blld, nd145):
    for a in blld.articles + nd145.articles:
        assert a.article_title, a.heading
        assert not a.article_title.endswith((".", ";", ":")), a.heading
        assert "  " not in a.article_title


def test_probation_article_structure(blld):
    a = blld.articles[24]
    assert a.article_no == 25
    assert (a.chapter_no, a.chapter_title) == ("III", "HỢP ĐỒNG LAO ĐỘNG")
    assert (a.section_no, a.section_title) == (1, "GIAO KẾT HỢP ĐỒNG LAO ĐỘNG")
    assert a.intro.startswith("Thời gian thử việc do hai bên thỏa thuận")
    assert [c.no for c in a.clauses] == [1, 2, 3, 4]
    assert a.clauses[0].text.startswith("Không quá 180 ngày")
    assert a.clauses[3].text == "Không quá 06 ngày làm việc đối với công việc khác."


def test_annual_leave_points(blld):
    a = blld.articles[112]
    assert a.article_no == 113
    first = a.clauses[0]
    assert [p.letter for p in first.points] == ["a", "b", "c"]
    assert first.points[0].text.startswith("12 ngày làm việc")
    assert first.points[2].text.startswith("16 ngày làm việc")
    assert len(a.clauses) == 7


def test_retirement_age_keeps_second_paragraph_in_clause(blld):
    clause2 = blld.articles[168].clauses[1]
    assert "62 tuổi đối với lao động nam vào năm 2028" in clause2.text
    assert "\nKể từ năm 2021" in clause2.text  # indented paragraph inside the clause


def test_quoted_amendments_are_not_structure(blld):
    # Điều 219 quotes "Điều 54." of the Social Insurance Law; it must stay inside Điều 219.
    a = blld.articles[218]
    assert a.article_no == 219
    assert [c.no for c in a.clauses] == [1, 2]
    assert "“Điều 54. Điều kiện hưởng lương hưu" in a.clauses[0].points[0].text


def test_clauses_and_points_are_sequential_everywhere(blld, nd145):
    for a in blld.articles + nd145.articles:
        assert [c.no for c in a.clauses] == list(range(1, len(a.clauses) + 1)), a.heading
        for points in [a.intro_points] + [c.points for c in a.clauses]:
            letters = [p.letter for p in points]
            assert letters == list(POINT_LETTERS[: len(letters)]), a.heading


def test_point_letter_dd_follows_d(nd145):
    # Vietnamese lettering: a, b, c, d, đ, e, ... "đ" must be recognised as a point.
    assert any(
        [p.letter for p in c.points][:6] == ["a", "b", "c", "d", "đ", "e"]
        for a in nd145.articles
        for c in a.clauses
    )


def test_no_parser_anomalies(blld, nd145):
    assert blld.anomalies == []
    assert nd145.anomalies == []


def test_decree_has_115_articles(nd145):
    assert [a.article_no for a in nd145.articles] == list(range(1, 116))
    assert nd145.articles[0].article_title == "Phạm vi điều chỉnh"
    assert nd145.articles[-1].article_title == "Trách nhiệm thi hành"


def test_decree_chapter_numbering_follows_the_source(nd145):
    """The decree prints "Chương XI" for both chapter 11 and the final chapter (checked
    against the signed scan). We keep the printed number and disambiguate with chapter_seq.
    The repeated chapter heading at the start of the second gazette issue is a continuation,
    not a new chapter."""
    by_no = {a.article_no: a for a in nd145.articles}
    assert (by_no[101].chapter_seq, by_no[101].chapter_no) == (11, "XI")
    assert (by_no[102].chapter_seq, by_no[102].chapter_no) == (11, "XI")
    assert by_no[102].section_no == 2
    last = by_no[114]
    assert (last.chapter_seq, last.chapter_no, last.chapter_title) == (
        12,
        "XI",
        "ĐIỀU KHOẢN THI HÀNH",
    )
    assert last.section_no is None


def test_decree_appendices(nd145):
    kinds = {a.appendix_no: a.kind for a in nd145.appendices}
    assert kinds == {
        "I": "forms", "II": "list", "III": "forms", "IV": "forms", "V": "forms", "VI": "list",
    }  # fmt: skip
    leasing = nd145.appendices[1]
    assert leasing.title == "DANH MỤC CÔNG VIỆC ĐƯỢC THỰC HIỆN CHO THUÊ LẠI LAO ĐỘNG"
    rows = leasing.text.split("\n")
    assert len(rows) == 20
    assert rows[0] == "1. Phiên dịch/Biên dịch/Tốc ký"
    assert rows[16] == "17. Lái xe"
    forms = nd145.appendices[4].text.split("\n")
    assert forms[0] == "Mẫu số 01/PLV: Hợp đồng lao động giúp việc gia đình"
    assert len(forms) == 3
