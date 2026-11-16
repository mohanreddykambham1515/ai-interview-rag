from pathlib import Path
import pymupdf


def load_pdf(file_path: str) -> list[dict]:
    documents =[]

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"pdf not found: {pdf_path}")

    pdf = pymupdf.open(pdf_path)

    try:
        for page_number , page in enumerate(pdf,start = 1):
            text = page.get_text('text').strip()

            if not text:
                continue

            documents.append({
                'content': text,
                'metadata': {
                    'source': pdf_path.name,
                    'page': page_number,
                    'total_pages': len(pdf)
                }
            })

    finally:
        pdf.close()


    return documents

def load_all_pdfs(directory: str) -> list[dict]:
    all_documents = []

    pdf_files = sorted(Path(directory).glob('*.pdf'))

    for pdf_file in pdf_files:
        documents = load_pdf(str(pdf_file))

        all_documents.extend(documents)
        print(
            f"loaded {pdf_file.name}"
            f"{len(documents)} pages"
        )
    return all_documents
