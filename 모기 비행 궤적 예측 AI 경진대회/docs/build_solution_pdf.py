"""build_solution_pdf — docs/SOLUTION.md 를 인쇄용 HTML로 변환.

PDF 생성: 변환된 SOLUTION.html 을 브라우저(헤드리스 Edge/Chrome)로 인쇄.
  msedge --headless --disable-gpu --print-to-pdf=SOLUTION.pdf --no-margins SOLUTION.html

사용: python docs/build_solution_pdf.py   (docs/SOLUTION.html 생성)
의존: pip install markdown
"""
from __future__ import annotations
import sys
from pathlib import Path
import markdown

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
SRC = HERE / "SOLUTION.md"
OUT = HERE / "SOLUTION.html"

CSS = """
@page { size: A4; margin: 18mm 16mm 16mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: 'Malgun Gothic', '맑은 고딕', 'Apple SD Gothic Neo', sans-serif;
  font-size: 10.5pt; line-height: 1.6; color: #1a1a1a; max-width: 860px;
  margin: 0 auto; padding: 8px 4px;
}
h1 { font-size: 20pt; border-bottom: 3px solid #2563eb; padding-bottom: 8px; margin: 4px 0 14px; }
h2 { font-size: 14.5pt; border-bottom: 1px solid #d1d5db; padding-bottom: 5px; margin: 24px 0 10px; color: #111827; }
h3 { font-size: 12pt; margin: 16px 0 6px; color: #1f2937; }
h2, h3, h4 { page-break-after: avoid; }
p, li { margin: 5px 0; }
blockquote { border-left: 4px solid #93c5fd; background: #f0f7ff; margin: 10px 0; padding: 6px 14px; color: #334155; }
code { font-family: 'D2Coding', 'Consolas', 'Courier New', monospace; background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 9.5pt; }
pre { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 12px; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 9pt; line-height: 1.45; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; page-break-inside: avoid; }
th, td { border: 1px solid #d1d5db; padding: 5px 9px; text-align: left; vertical-align: top; }
th { background: #eff6ff; font-weight: 700; }
tr:nth-child(even) td { background: #fafbfc; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 20px 0; }
strong { color: #b91c1c; }
a { color: #2563eb; text-decoration: none; }
 table strong { color: #111827; }
"""


def main():
    md = SRC.read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists", "nl2br"])
    html = (f"<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'>"
            f"<title>모기 비행 궤적 예측 솔루션</title><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")
    OUT.write_text(html, encoding="utf-8")
    print(f"[saved] {OUT}  ({len(html)} bytes)")
    print("PDF: msedge --headless --disable-gpu --print-to-pdf 로 인쇄 (build 스크립트 주석 참고)")


if __name__ == "__main__":
    main()
