import pymupdf
from pathlib import Path

def extract_text_from_bytes(pdf_bytes):

    pages = []

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as document:

        for page_number, page in enumerate(document):

            text = page.get_text()

            pages.append({
                "page": page_number + 1,
                "text": text
            })

    return pages
    
def extract_text_from_pdf(pdf_path):
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    document = pymupdf.open(pdf_path)
    pages_text = []

    for page_number, page in enumerate(document):
        text = page.get_text()
        pages_text.append({
            "page": page_number +1,
            "text": text
        })

    document.close()
    return pages_text