"""Registry of the legal documents in the corpus and where to download them.

Text source: the Official Gazette (Công báo) PDFs on congbaocdn.chinhphu.vn. They carry a
real text layer, while the signed PDFs on vanban.chinhphu.vn are image scans. Texts published
in Công báo are official and carry the same value as the original document.
See docs/decisions/001-corpus-source-and-pdf-extraction.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceFile:
    url: str
    filename: str
    sha256: str  # pinned at first download; fetch warns when the upstream file changes
    note: str = ""


@dataclass(frozen=True)
class SourceDocument:
    doc_short: str  # stable short id used in chunk ids and URLs, e.g. "BLLD2019"
    doc_number: str  # official number (số hiệu), e.g. "45/2019/QH14"
    short_title: str  # used in chunk headers and citations, e.g. "Bộ luật Lao động 2019"
    title: str
    doc_type: str
    issued_date: str  # ISO date the document was signed/passed
    effective_date: str
    source_url: str  # human-friendly official page, shown in citations
    gazette_url: str
    files: tuple[SourceFile, ...]
    # Regexes (matched against a stripped line) that mark the end of the normative body:
    # the signature block. Everything after it is appendices or noise.
    body_end_patterns: tuple[str, ...] = field(default_factory=tuple)

    @property
    def display_name(self) -> str:
        return f"{self.title} số {self.doc_number}"


BLLD2019 = SourceDocument(
    doc_short="BLLD2019",
    doc_number="45/2019/QH14",
    short_title="Bộ luật Lao động 2019",
    title="Bộ luật Lao động",
    doc_type="Bộ luật",
    issued_date="2019-11-20",
    effective_date="2021-01-01",
    source_url="https://vanban.chinhphu.vn/?pageid=27160&docid=198540",
    gazette_url="https://congbao.chinhphu.vn/van-ban/nghi-quyet-so-45-2019-qh14-30232.htm",
    files=(
        SourceFile(
            url="https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2019/11/30232/29070-1-2019993-99445-2019-qh14.pdf",
            filename="BLLD2019_congbao_993-994.pdf",
            sha256="3c865f3d92ba23604e316360b902297ddbc0d1c394d3c08a466029a9d43b5b6c",
            note="Công báo số 993 + 994, ngày 26-12-2019",
        ),
    ),
    body_end_patterns=(r"^Bộ luật này được Quốc hội", r"^CHỦ TỊCH QUỐC HỘI$"),
)

ND145 = SourceDocument(
    doc_short="ND145",
    doc_number="145/2020/NĐ-CP",
    short_title="Nghị định 145/2020/NĐ-CP",
    title="Nghị định quy định chi tiết và hướng dẫn thi hành một số điều của Bộ luật Lao động "
    "về điều kiện lao động và quan hệ lao động",
    doc_type="Nghị định",
    issued_date="2020-12-14",
    effective_date="2021-02-01",
    source_url="https://vanban.chinhphu.vn/?pageid=27160&docid=201967",
    gazette_url="https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-145-2020-nd-cp-32732.htm",
    files=(
        SourceFile(
            url="https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2020/12/32732/33806-1-20201203-1204145-2020-nd-cp.pdf",
            filename="ND145_congbao_1203-1204.pdf",
            sha256="3131bee5778b06390f4f62fcffb751fdd78bf14f1a834ae17e0f9c1f319ca65a",
            note="Công báo số 1203 + 1204, ngày 28-12-2020 (Điều 1 to Điều 101)",
        ),
        SourceFile(
            url="https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2020/12/32732/33809-1-20201205-1206145-2020-nd-cp.pdf",
            filename="ND145_congbao_1205-1206.pdf",
            sha256="38de13854df47d0e27ee497b86272f874742cea7e68f1be26d81587ff95f170b",
            note="Công báo số 1205 + 1206, ngày 28-12-2020 (Điều 102 to Điều 115, appendices)",
        ),
    ),
    body_end_patterns=(r"^TM\. CHÍNH PHỦ$",),
)

DOCUMENTS: tuple[SourceDocument, ...] = (BLLD2019, ND145)


def get_document(doc_short: str) -> SourceDocument:
    for doc in DOCUMENTS:
        if doc.doc_short.lower() == doc_short.lower():
            return doc
    raise KeyError(f"Unknown document {doc_short!r}; known: {[d.doc_short for d in DOCUMENTS]}")
