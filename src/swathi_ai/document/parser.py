from pathlib import Path
from pypdf import PdfReader
from docx import Document


def read_pdf(path: Path):
    reader = PdfReader(path)

    pages = []

    for page in reader.pages:
        pages.append(page.extract_text() or "")

    return pages


def read_docx(path: Path):
    doc = Document(path)

    text = "\n".join(p.text for p in doc.paragraphs)

    return [text]


def read_txt(path: Path):
    return [path.read_text(encoding="utf-8")]