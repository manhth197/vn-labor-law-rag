"""Drive the chat UI in the installed Microsoft Edge and save screenshots for the README.

Requires a running server (`python -m vnlaw_rag.cli serve`) and `pip install playwright`.
It never downloads a browser: it uses the system Edge (`channel="msedge"`). Each question
costs at most one Gemini request (plus a fallback when a model is busy); the out-of-scope
question normally stops at the `grade` node without any request.

    python scripts/screenshots.py --base-url http://127.0.0.1:8000 --out docs/screenshots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QUESTIONS = [
    ("answer", "Thời gian thử việc tối đa đối với công việc cần trình độ cao đẳng là bao lâu?"),
    ("holiday-warning", "Người lao động được nghỉ Tết Âm lịch mấy ngày?"),
    ("out-of-scope", "Lãi suất vay mua nhà ở xã hội năm nay là bao nhiêu?"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="docs/screenshots")
    parser.add_argument("--timeout", type=float, default=90.0, help="Seconds per answer.")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 900}, locale="vi-VN")
        page.goto(args.base_url)
        page.wait_for_selector("#question")
        for index, (name, question) in enumerate(QUESTIONS):
            page.fill("#question", question)
            page.click("#send")
            answer = page.locator("article.answer").nth(index)
            # the meta line is written only when the final `result` event has been rendered
            meta = answer.locator(".meta")
            error = answer.locator(".text.error")
            page.wait_for_function(
                """(i) => {
                    const a = document.querySelectorAll('article.answer')[i];
                    return a && (a.querySelector('.meta').textContent.trim() ||
                                 a.querySelector('.text.error'));
                }""",
                arg=index,
                timeout=args.timeout * 1000,
            )
            path = out / f"{index + 1}-{name}.png"
            answer.screenshot(path=str(path))
            report.append(
                {
                    "question": question,
                    "meta": meta.inner_text(),
                    "error": error.inner_text() if error.count() else None,
                    "answer": answer.locator(".text").inner_text()[:300],
                    "warnings": answer.locator(".warnings li").all_inner_texts(),
                    "citations": answer.locator(".cite").all_inner_texts(),
                    "screenshot": str(path),
                }
            )
        page.screenshot(path=str(out / "0-overview.png"), full_page=True)
        browser.close()
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
