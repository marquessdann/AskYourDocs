from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import UnsupportedFileError


@dataclass(frozen=True)
class PageText:
    page_number: int  # 1-indexed, matches how a human would cite the page
    text: str


def extract_pages(file_bytes: bytes) -> list[PageText]:
    """Extracts raw text per page from a PDF. Pages with no extractable text
    (e.g. scanned images without OCR) are skipped rather than stored empty."""
    try:
        reader = PdfReader(BytesIO(file_bytes))
    except PdfReadError as exc:
        raise UnsupportedFileError(f"Could not read PDF: {exc}") from exc

    if reader.is_encrypted:
        raise UnsupportedFileError("Encrypted PDFs are not supported.")

    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(PageText(page_number=index, text=text))

    if not pages:
        raise UnsupportedFileError("No extractable text found in the PDF (is it a scanned image?).")

    return pages
