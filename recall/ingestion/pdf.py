from __future__ import annotations

from dataclasses import dataclass

import fitz


class PdfExtractionError(ValueError):
    """Raised when a PDF cannot produce trustworthy extracted text."""


@dataclass(frozen=True)
class ExtractedPdfPage:
    page_number: int
    text: str


def extract_pdf_pages(pdf_bytes: bytes) -> list[ExtractedPdfPage]:
    if not pdf_bytes:
        raise PdfExtractionError("Cannot ingest an empty file.")

    if not pdf_bytes.lstrip().startswith(b"%PDF"):
        raise PdfExtractionError("Only PDF files can be ingested.")

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PdfExtractionError("The PDF could not be opened.") from exc

    with document:
        if document.is_encrypted:
            raise PdfExtractionError("Encrypted PDFs are not supported.")

        pages: list[ExtractedPdfPage] = []
        for zero_based_index in range(document.page_count):
            page = document.load_page(zero_based_index)
            text = normalize_text(page.get_text("text"))
            pages.append(
                ExtractedPdfPage(
                    page_number=zero_based_index + 1,
                    text=text,
                )
            )

    if not pages:
        raise PdfExtractionError("The PDF does not contain any pages.")

    if not any(has_meaningful_text(page.text) for page in pages):
        raise PdfExtractionError("The PDF has no meaningful extractable text.")

    return pages


def normalize_text(text: str) -> str:
    normalized_lines = (
        line.rstrip() for line in text.replace("\r\n", "\n").splitlines()
    )
    return "\n".join(normalized_lines).strip()


def has_meaningful_text(text: str) -> bool:
    return any(character.isalnum() for character in text)
