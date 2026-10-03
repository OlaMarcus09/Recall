from __future__ import annotations

import pytest

import fitz

from recall.db import (
    current_schema_version,
    initialize_database,
    list_material_pages,
    material_count,
    page_count,
)
from recall.ingestion.pdf import PdfExtractionError
from recall.ingestion.service import ingest_pdf


def make_pdf(page_texts: list[str]) -> bytes:
    document = fitz.open()
    try:
        for page_text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), page_text)
        return document.tobytes()
    finally:
        document.close()


def test_ingests_material_metadata_and_page_provenance(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    initialize_database(database_path)
    pdf_bytes = make_pdf(
        [
            "Synthetic page one: introduction to trustworthy memory.",
            "Synthetic page two: page ordering remains deterministic.",
            "Synthetic page three: final provenance sample.",
        ]
    )

    result = ingest_pdf(
        database_path=database_path,
        pdf_bytes=pdf_bytes,
        original_filename="../unsafe/course.pdf",
        title="Synthetic Course",
    )

    assert result.created is True
    assert result.material.title == "Synthetic Course"
    assert result.material.original_filename == "course.pdf"
    assert result.material.source_type == "pdf"
    assert result.material.page_count == 3
    assert result.material.content_sha256
    assert [page.material_id for page in result.pages] == [result.material.id] * 3
    assert [page.page_number for page in result.pages] == [1, 2, 3]
    assert "page one" in result.pages[0].text
    assert "page three" in result.pages[2].text


def test_pages_are_persisted_in_source_order(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    initialize_database(database_path)
    pdf_bytes = make_pdf(["Alpha first page.", "Beta second page.", "Gamma third page."])

    result = ingest_pdf(
        database_path=database_path,
        pdf_bytes=pdf_bytes,
        original_filename="ordered.pdf",
    )

    persisted_pages = list_material_pages(database_path, result.material.id)
    assert [page["page_number"] for page in persisted_pages] == [1, 2, 3]
    assert "Alpha" in persisted_pages[0]["text"]
    assert "Beta" in persisted_pages[1]["text"]
    assert "Gamma" in persisted_pages[2]["text"]


def test_duplicate_ingestion_returns_existing_material(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    initialize_database(database_path)
    pdf_bytes = make_pdf(["A stable PDF should not duplicate its material."])

    first = ingest_pdf(
        database_path=database_path,
        pdf_bytes=pdf_bytes,
        original_filename="first.pdf",
        title="First Title",
    )
    second = ingest_pdf(
        database_path=database_path,
        pdf_bytes=pdf_bytes,
        original_filename="renamed.pdf",
        title="Second Title",
    )

    assert first.created is True
    assert second.created is False
    assert second.material.id == first.material.id
    assert second.material.title == "First Title"
    assert material_count(database_path) == 1
    assert page_count(database_path) == 1


@pytest.mark.parametrize(
    ("pdf_bytes", "message"),
    [
        (b"", "empty file"),
        (b"not a pdf", "Only PDF files"),
        (b"%PDF-1.7\nthis is not a complete pdf", "could not be opened"),
    ],
)
def test_invalid_inputs_fail_without_partial_records(tmp_path, pdf_bytes, message):
    database_path = tmp_path / "recall.sqlite3"
    initialize_database(database_path)

    with pytest.raises(PdfExtractionError, match=message):
        ingest_pdf(
            database_path=database_path,
            pdf_bytes=pdf_bytes,
            original_filename="bad.pdf",
        )

    assert material_count(database_path) == 0
    assert page_count(database_path) == 0


def test_pdf_with_no_meaningful_text_fails_without_partial_records(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    initialize_database(database_path)
    document = fitz.open()
    try:
        document.new_page()
        pdf_bytes = document.tobytes()
    finally:
        document.close()

    with pytest.raises(PdfExtractionError, match="no meaningful extractable text"):
        ingest_pdf(
            database_path=database_path,
            pdf_bytes=pdf_bytes,
            original_filename="blank.pdf",
        )

    assert material_count(database_path) == 0
    assert page_count(database_path) == 0


def test_schema_version_is_upgraded_for_ingestion_tables(tmp_path):
    database_path = tmp_path / "recall.sqlite3"

    initialize_database(database_path)

    assert current_schema_version(database_path) == 3
