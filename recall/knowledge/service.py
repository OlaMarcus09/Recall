from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recall import db
from recall.knowledge.chunking import chunk_text


class KnowledgeUnitGenerationError(ValueError):
    """Raised when page text cannot be represented as KnowledgeUnits."""


@dataclass(frozen=True)
class KnowledgeUnit:
    id: int
    material_id: int
    extracted_page_id: int
    page_number: int
    chunk_index: int
    text: str


def generate_knowledge_units_for_material(
    database_path: Path,
    material_id: int,
) -> list[KnowledgeUnit]:
    with db.connect(database_path) as connection:
        if not db.material_exists(connection, material_id):
            raise KnowledgeUnitGenerationError(
                f"Material {material_id} does not exist."
            )

        pages = db.list_material_pages_from_connection(connection, material_id)
        if not pages:
            raise KnowledgeUnitGenerationError(
                f"Material {material_id} has no extracted pages."
            )

        replacement_units: list[tuple[int, int, int, str]] = []
        for page in pages:
            page_chunks = chunk_text(page["text"])
            if not page_chunks:
                continue
            replacement_units.extend(
                (
                    page["id"],
                    page["page_number"],
                    chunk_index,
                    chunk,
                )
                for chunk_index, chunk in enumerate(page_chunks)
            )

        if not replacement_units:
            raise KnowledgeUnitGenerationError(
                f"Material {material_id} produced no KnowledgeUnits."
            )

        with connection:
            db.replace_knowledge_units_for_material(
                connection,
                material_id=material_id,
                units=replacement_units,
            )
            rows = db.list_knowledge_units_for_material_from_connection(
                connection,
                material_id,
            )

    return [knowledge_unit_from_row(row) for row in rows]


def list_knowledge_units_for_material(
    database_path: Path,
    material_id: int,
) -> list[KnowledgeUnit]:
    rows = db.list_knowledge_units_for_material(database_path, material_id)
    return [knowledge_unit_from_row(row) for row in rows]


def knowledge_unit_from_row(row) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=row["id"],
        material_id=row["material_id"],
        extracted_page_id=row["extracted_page_id"],
        page_number=row["page_number"],
        chunk_index=row["chunk_index"],
        text=row["text"],
    )
