"""Prompts and the structured-output schema for the `generate` node (Vietnamese, as the
answers are)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

SYSTEM_PROMPT = """\
Bạn là trợ lý tra cứu pháp luật lao động Việt Nam. Bạn chỉ được trả lời dựa trên các NGUỒN \
được cung cấp trong mỗi câu hỏi (trích từ Bộ luật Lao động 2019 theo văn bản hợp nhất \
125/VBHN-VPQH và Nghị định 145/2020/NĐ-CP).

Quy tắc:
1. Trả lời bằng tiếng Việt, ngắn gọn, chính xác, đúng câu chữ của văn bản khi nêu con số, \
thời hạn, tỷ lệ.
2. Mỗi ý phải nêu rõ Điều, khoản (ví dụ: "khoản 2 Điều 25 Bộ luật Lao động") và gắn mã nguồn \
trong ngoặc vuông ngay sau ý đó, ví dụ [BLLD2019-D25].
3. Chỉ dùng mã nguồn có trong danh sách NGUỒN. Không bịa Điều, khoản, con số hay mã nguồn.
4. Nếu nguồn được trích dẫn có mục "Lưu ý thay đổi", phải nêu lưu ý đó trong câu trả lời, \
kèm ngày hiệu lực.
5. Nếu các nguồn không đủ để trả lời, đặt sufficient = false và nói rõ là không tìm thấy quy \
định phù hợp trong các văn bản này.
6. Không tư vấn cho một vụ việc cụ thể; chỉ nêu quy định.
7. Liệt kê trong `citations` mọi mã nguồn đã dùng."""


class GeneratedAnswer(BaseModel):
    """Structured output requested from the model (kept small: schema size is limited)."""

    answer: str = Field(description="Câu trả lời tiếng Việt, có mã nguồn trong ngoặc vuông.")
    citations: list[str] = Field(description="Các mã nguồn đã dùng, ví dụ BLLD2019-D25.")
    sufficient: bool = Field(description="false nếu các nguồn không đủ để trả lời.")


def source_label(source: dict[str, Any]) -> str:
    if source.get("unit") == "appendix":
        return f"{source['doc_title']} - Phụ lục {source['appendix_no']}"
    clauses = source.get("clause_nos") or []
    part = f" (khoản {clauses[0]}-{clauses[-1]})" if len(clauses) > 1 else ""
    part = f" (khoản {clauses[0]})" if len(clauses) == 1 else part
    return f"{source['doc_title']} - Điều {source['article_no']}. {source['article_title']}{part}"


def build_prompt(question: str, sources: Sequence[dict[str, Any]]) -> str:
    blocks = []
    for s in sources:
        lines = [f"[{s['chunk_id']}] {source_label(s)}", s["content"]]
        notes = [*s.get("amended_by", []), *s.get("upcoming_change", [])]
        lines.extend(f"Lưu ý thay đổi: {note}" for note in notes)
        blocks.append("\n".join(lines))
    return f"CÂU HỎI: {question}\n\nNGUỒN:\n\n" + "\n\n---\n\n".join(blocks)
