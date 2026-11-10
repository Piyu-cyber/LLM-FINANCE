import os
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image

def render_and_inspect_pdf(pdf_path, output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pdf = pdfium.PdfDocument(str(Path(pdf_path).resolve()))
    num_pages = len(pdf)
    print(f"Loaded PDF: {pdf_path} with {num_pages} pages.")
    
    page_stats = []
    
    for i in range(num_pages):
        page = pdf[i]
        # Render at 150 DPI for crisp visual inspection
        bitmap = page.render(scale=150/72)
        pil_img = bitmap.to_pil()
        
        img_name = f"page_{i+1}.png"
        img_path = out_dir / img_name
        pil_img.save(str(img_path))
        
        width, height = pil_img.size
        print(f"Page {i+1}: Rendered to {img_name} ({width} x {height} px)")
        
        page_stats.append({
            "page_num": i + 1,
            "width": width,
            "height": height,
            "img_path": str(img_path)
        })
        
    return page_stats

if __name__ == "__main__":
    pdf_file = "submission/LLM-FINANCE_Project_Report.pdf"
    pages_dir = "submission/pdf_pages"
    render_and_inspect_pdf(pdf_file, pages_dir)
