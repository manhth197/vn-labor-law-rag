"""Human review of the golden set through an Excel workbook.

`export_review` writes a stratified sample (same type proportions as the full set, seeded so
it is reproducible) with the expected articles and the first 300 characters of each, so the
owner can judge an item without opening the law. `import_review` reads the filled workbook
back into golden.jsonl:

- "Đúng": reviewed = true; edits made in the question / references / answer cells are kept.
- "Cần sửa": edits in those cells are applied. If something was edited the item becomes
  reviewed (status "fixed"); if only a note was written it stays unreviewed with status
  "needs_fix" so the note is followed up.
- "Bỏ": the item is removed from golden.jsonl and appended to golden_removed.jsonl.
- empty: untouched.

openpyxl is a dev dependency and imported lazily.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from vnlaw_rag.eval.golden import GoldenItem, Ref, UnitKey, load_golden, write_golden

DOC_LABELS = {"BLLD2019": "BLLĐ 2019", "ND145": "NĐ 145/2020"}
LABEL_DOCS = {v: k for k, v in DOC_LABELS.items()}
TYPE_LABELS = {
    "lookup": "tra cứu",
    "paraphrase": "diễn đạt lại",
    "numeric": "con số",
    "multi": "nhiều điều",
    "unanswerable": "ngoài phạm vi",
}
VERDICTS = ("Đúng", "Cần sửa", "Bỏ")
NO_REFS = "(không có: câu hỏi ngoài phạm vi)"
HEADERS = [
    ("id", 8),
    ("Loại", 14),
    ("Câu hỏi", 45),
    ("Điều kỳ vọng (văn bản, số Điều, tiêu đề)", 40),
    ("Trích 300 ký tự đầu của Điều", 70),
    ("Câu trả lời tham chiếu", 55),
    ("Đánh giá", 12),
    ("Ghi chú", 40),
]
_REF_RE = re.compile(
    r"(BLLĐ 2019|NĐ 145/2020)\s*[-–:,]?\s*(?:Điều\s+(\d+)|Phụ lục\s+([IVXLC]+))", re.IGNORECASE
)


def stratified_sample(items: list[GoldenItem], n: int, seed: int = 42) -> list[GoldenItem]:
    """Pick n items keeping each type's share (largest remainder rounding)."""
    by_type: dict[str, list[GoldenItem]] = defaultdict(list)
    for item in items:
        by_type[item.type].append(item)
    quotas = {t: n * len(v) / len(items) for t, v in by_type.items()}
    counts = {t: math.floor(q) for t, q in quotas.items()}
    for t in sorted(quotas, key=lambda t: (counts[t] - quotas[t], t))[: n - sum(counts.values())]:
        counts[t] += 1
    rng = random.Random(seed)
    picked = {i.id for t, group in by_type.items() for i in rng.sample(group, counts[t])}
    return [i for i in items if i.id in picked]


def format_refs(refs: list[Ref], titles: dict[UnitKey, str]) -> str:
    if not refs:
        return NO_REFS
    lines = []
    for ref in refs:
        unit = f"Điều {ref.article}" if ref.article is not None else f"Phụ lục {ref.appendix}"
        lines.append(f"{DOC_LABELS[ref.doc]} - {unit}: {titles.get(ref.key(), '?')}")
    return "\n".join(lines)


def parse_refs(cell: str) -> list[Ref]:
    refs = []
    for doc_label, article, appendix in _REF_RE.findall(cell or ""):
        doc = next(v for k, v in LABEL_DOCS.items() if k.lower() == doc_label.lower())
        if article:
            refs.append(Ref(doc=doc, article=int(article)))
        else:
            refs.append(Ref(doc=doc, appendix=appendix.upper()))
    return refs


def export_review(
    golden_path: Path,
    texts: dict[UnitKey, str],
    titles: dict[UnitKey, str],
    out_path: Path,
    n: int = 25,
    seed: int = 42,
) -> list[GoldenItem]:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    items = load_golden(golden_path)
    sample = stratified_sample(items, n, seed)
    wb = Workbook()
    ws = wb.active
    ws.title = "Duyệt"
    ws.append([h for h, _ in HEADERS])
    for col, (_, width) in enumerate(HEADERS, start=1):
        ws.column_dimensions[ws.cell(row=1, column=col).column_letter].width = width
        cell = ws.cell(row=1, column=col)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDEBF7")
    for item in sample:
        excerpt = "\n\n".join(texts[r.key()][:300] + "…" for r in item.expected) or NO_REFS
        ws.append(
            [
                item.id,
                TYPE_LABELS[item.type],
                item.question,
                format_refs(item.expected, titles),
                excerpt,
                item.reference_answer,
                "",
                "",
            ]
        )
    wrap = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
        for cell in row:
            cell.alignment = wrap
    validation = DataValidation(
        type="list", formula1='"' + ",".join(VERDICTS) + '"', allow_blank=True
    )
    validation.error = "Chọn Đúng, Cần sửa hoặc Bỏ"
    ws.add_data_validation(validation)
    validation.add(f"G2:G{ws.max_row}")
    ws.freeze_panes = "C2"

    guide = wb.create_sheet("Hướng dẫn")
    for line in [
        "Cách duyệt bộ câu hỏi đánh giá (golden set)",
        "",
        f"1. Mỗi dòng ở trang 'Duyệt' là một câu hỏi ({len(sample)} câu, chọn ngẫu nhiên "
        "theo tỉ lệ từng loại).",
        "2. Kiểm tra: câu hỏi có tự nhiên không; Điều kỳ vọng có đúng là nơi trả lời không; "
        "câu trả lời tham chiếu có đúng và đủ không.",
        "3. Cột 'Đánh giá': chọn Đúng / Cần sửa / Bỏ.",
        "4. Nếu Cần sửa: sửa trực tiếp vào ô Câu hỏi, Điều kỳ vọng hoặc Câu trả lời tham chiếu. "
        "Điều kỳ vọng ghi theo mẫu 'BLLĐ 2019 - Điều 25' hoặc 'NĐ 145/2020 - Phụ lục II', "
        "mỗi Điều một dòng; phần tiêu đề sau dấu ':' có thể bỏ.",
        "5. Nếu chỉ ghi vào 'Ghi chú' mà không sửa ô nào, câu đó được đánh dấu cần sửa tiếp.",
        "6. 'Bỏ': câu hỏi bị loại khỏi bộ đánh giá.",
        "",
        "Dữ liệu là văn bản gốc BLLĐ 45/2019/QH14 và NĐ 145/2020/NĐ-CP; chưa phản ánh sửa đổi.",
    ]:
        guide.append([line])
    guide.column_dimensions["A"].width = 120
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    return sample


@dataclass
class ImportSummary:
    counts: Counter = field(default_factory=Counter)
    removed: list[str] = field(default_factory=list)
    edited: list[str] = field(default_factory=list)


def import_review(xlsx_path: Path, golden_path: Path, removed_path: Path) -> ImportSummary:
    from openpyxl import load_workbook

    items = {i.id: i for i in load_golden(golden_path)}
    ws = load_workbook(xlsx_path, read_only=True)["Duyệt"]
    summary = ImportSummary()
    removed: list[GoldenItem] = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row or not row[0]:
            continue
        item_id, _, question, refs_cell, _, answer, verdict, note = [*row, *[None] * 8][:8]
        verdict = (verdict or "").strip()
        if item_id not in items or not verdict:
            summary.counts["skipped"] += 1
            continue
        if verdict not in VERDICTS:
            raise ValueError(f"{item_id}: unknown verdict {verdict!r}")
        item = items[item_id]
        if verdict == "Bỏ":
            removed.append(items.pop(item_id))
            summary.removed.append(item_id)
            summary.counts["removed"] += 1
            continue
        edited = _apply_edits(item, question, refs_cell, answer)
        if edited:
            summary.edited.append(item_id)
        item.review_note = (note or "").strip() or None
        if verdict == "Đúng" or edited:
            item.reviewed, item.review_status = True, ("fixed" if edited else "ok")
        else:
            item.reviewed, item.review_status = False, "needs_fix"
        summary.counts[item.review_status] += 1

    write_golden(golden_path, items.values())
    if removed:
        with removed_path.open("a", encoding="utf-8", newline="\n") as fh:
            for item in removed:
                fh.write(item.model_dump_json(exclude_none=True) + "\n")
    return summary


def _apply_edits(item: GoldenItem, question, refs_cell, answer) -> bool:
    edited = False
    if question and question.strip() != item.question:
        item.question, edited = question.strip(), True
    if answer and answer.strip() != item.reference_answer:
        item.reference_answer, edited = answer.strip(), True
    if refs_cell is not None and refs_cell.strip() != NO_REFS:
        refs = parse_refs(refs_cell)
        if refs and [r.key() for r in refs] != [r.key() for r in item.expected]:
            item.expected, edited = refs, True
    return edited
