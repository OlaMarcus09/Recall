from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePath

from recall import db
from recall.ingestion.pdf import extract_pdf_pages


SOURCE_TYPE_PDF = "pdf"


@dataclass(frozen=True)
class Material:
    id: int
    title: str
    original_filename: str
    source_type: str
    uploaded_at: str
    content_sha256: str
    page_count: int


@dataclass(frozen=True)
class IngestedPage:
    id: int
    material_id: int
    page_number: int
    text: str


@dataclass(frozen=True)
class IngestionResult:
    material: Material
    pages: list[IngestedPage]
    created: bool


def ingest_pdf(
    *,
    database_path,
    pdf_bytes: bytes,
    original_filename: str,
    title: str | None = None,
) -> IngestionResult:
    content_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    safe_filename = safe_display_filename(original_filename)
    material_title = safe_title(title, safe_filename)
    extracted_pages = extract_pdf_pages(pdf_bytes)

    with db.connect(database_path) as connection:
        existing = db.get_material_by_hash(connection, content_sha256)
        if existing is not None:
            material = material_from_row(existing)
            return IngestionResult(
                material=material,
                pages=load_pages(connection, material.id),
                created=False,
            )

        with connection:
            material_row = db.insert_material(
                connection,
                title=material_title,
                original_filename=safe_filename,
                source_type=SOURCE_TYPE_PDF,
                content_sha256=content_sha256,
                page_count_value=len(extracted_pages),
            )
            material = material_from_row(material_row)
            db.insert_extracted_pages(
                connection,
                material_id=material.id,
                pages=[
                    (page.page_number, page.text)
                    for page in extracted_pages
                ],
            )
            pages = load_pages(connection, material.id)

    return IngestionResult(material=material, pages=pages, created=True)


def safe_display_filename(original_filename: str) -> str:
    name = PurePath(original_filename).name.strip()
    return name or "uploaded.pdf"


def safe_title(title: str | None, filename: str) -> str:
    if title is not None and title.strip():
        return title.strip()
    if filename.lower().endswith(".pdf"):
        return filename[:-4] or filename
    return filename


def material_from_row(row) -> Material:
    return Material(
        id=row["id"],
        title=row["title"],
        original_filename=row["original_filename"],
        source_type=row["source_type"],
        uploaded_at=row["uploaded_at"],
        content_sha256=row["content_sha256"],
        page_count=row["page_count"],
    )


def page_from_row(row) -> IngestedPage:
    return IngestedPage(
        id=row["id"],
        material_id=row["material_id"],
        page_number=row["page_number"],
        text=row["text"],
    )


def load_pages(connection, material_id: int) -> list[IngestedPage]:
    rows = connection.execute(
        """
        SELECT id, material_id, page_number, text
        FROM extracted_pages
        WHERE material_id = ?
        ORDER BY page_number ASC
        """,
        (material_id,),
    ).fetchall()
    return [page_from_row(row) for row in rows]
