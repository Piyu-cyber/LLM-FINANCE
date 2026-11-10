import os
import sys
from pathlib import Path
import win32com.client

def convert_to_pdf(docx_path, pdf_path):
    abs_docx = str(Path(docx_path).resolve())
    abs_pdf = str(Path(pdf_path).resolve())
    
    print(f"Converting:\n  Input:  {abs_docx}\n  Output: {abs_pdf}")
    
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        doc = word.Documents.Open(abs_docx)
        # wdFormatPDF = 17
        doc.SaveAs(abs_pdf, FileFormat=17)
        doc.Close()
        print(f"Successfully converted to PDF: {abs_pdf} ({Path(abs_pdf).stat().st_size} bytes)")
    except Exception as e:
        print(f"Error during Word conversion: {e}", file=sys.stderr)
        raise
    finally:
        word.Quit()

if __name__ == "__main__":
    docx_file = "submission/LLM-FINANCE_Project_Report.docx"
    pdf_file = "submission/LLM-FINANCE_Project_Report.pdf"
    convert_to_pdf(docx_file, pdf_file)
