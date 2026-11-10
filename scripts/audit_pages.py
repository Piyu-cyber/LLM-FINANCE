import sys
import pypdfium2 as pdfium
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

pdf = pdfium.PdfDocument("submission/LLM-FINANCE_Project_Report.pdf")

for i, page in enumerate(pdf):
    textpage = page.get_textpage()
    text = textpage.get_text_range()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    first_few = " | ".join(lines[:3]) if lines else "EMPTY"
    last_few = " | ".join(lines[-2:]) if lines else "EMPTY"
    print(f"=== PAGE {i+1} ({len(lines)} lines) ===")
    print(f"  Start: {first_few[:120]}")
    print(f"  End:   {last_few[:120]}")

