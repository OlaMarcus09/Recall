from __future__ import annotations

import sqlite3

import pytest

from recall import db
from recall.db import (
    current_schema_version,
    initialize_database,
    knowledge_unit_count,
    list_knowledge_units_for_material,
)
from recall.ingestion.service import ingest_pdf
from recall.knowledge.chunking import MAX_CHARS, normalize_chunk_text
from recall.knowledge.service import (
    KnowledgeUnitGenerationError,
    generate_knowledge_units_for_material,
)
from tests.test_pdf_ingestion import make_pdf


def ingest_synthetic_material(database_path, page_texts, title="Synthetic Course"):
    initialize_database(database_path)
    return ingest_pdf(
        database_path=database_path,
        pdf_bytes=make_pdf(page_texts),
        original_filename="synthetic.pdf",
        title=title,
    )


def create_material_with_pages(database_path, page_texts, title="Synthetic Course"):
    initialize_database(database_path)
    with db.connect(database_path) as connection:
        with connection:
            material = db.insert_material(
                connection,
                title=title,
                original_filename=f"{title.lower().replace(' ', '-')}.pdf",
                source_type="pdf",
                content_sha256=f"synthetic-{title}-{len(page_texts)}",
                page_count_value=len(page_texts),
            )
            db.insert_extracted_pages(
                connection,
                material_id=material["id"],
                pages=[
                    (page_number, page_text)
                    for page_number, page_text in enumerate(page_texts, start=1)
                ],
            )
    return material


def test_generates_persistent_units_with_page_provenance(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    ingestion = ingest_synthetic_material(
        database_path,
        [
            "Page one introduces accountable memory.",
            "Page two preserves exact page provenance.",
            "Page three closes the safe fixture.",
        ],
    )

    units = generate_knowledge_units_for_material(
        database_path,
        ingestion.material.id,
    )

    assert len(units) == 3
    assert [unit.material_id for unit in units] == [ingestion.material.id] * 3
    assert [unit.extracted_page_id for unit in units] == [
        page.id for page in ingestion.pages
    ]
    assert [unit.page_number for unit in units] == [1, 2, 3]
    assert [unit.chunk_index for unit in units] == [0, 0, 0]
    assert "Page one" in units[0].text
    assert "Page three" in units[2].text


def test_ordering_is_page_then_chunk_index(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    long_page = " ".join(f"Sentence {index}." for index in range(160))
    material = create_material_with_pages(
        database_path,
        ["First page.", long_page, "Third page."],
    )

    generate_knowledge_units_for_material(database_path, material["id"])
    rows = list_knowledge_units_for_material(database_path, material["id"])

    ordering = [
        (row["page_number"], row["chunk_index"])
        for row in rows
    ]
    assert ordering == sorted(ordering)
    assert ordering[0] == (1, 0)
    assert ordering[-1][0] == 3


def test_long_page_produces_multiple_bounded_chunks(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    long_page = " ".join(
        f"This is a deterministic sentence number {index}."
        for index in range(120)
    )
    material = create_material_with_pages(database_path, [long_page])

    units = generate_knowledge_units_for_material(
        database_path,
        material["id"],
    )

    assert len(units) > 1
    assert all(len(unit.text) <= MAX_CHARS for unit in units)
    assert [unit.chunk_index for unit in units] == list(range(len(units)))


def test_chunks_never_cross_page_boundaries(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    ingestion = ingest_synthetic_material(
        database_path,
        [
            "PAGE_ONE_ONLY alpha alpha alpha.",
            "PAGE_TWO_ONLY beta beta beta.",
        ],
    )

    units = generate_knowledge_units_for_material(
        database_path,
        ingestion.material.id,
    )

    page_one_units = [unit for unit in units if unit.page_number == 1]
    page_two_units = [unit for unit in units if unit.page_number == 2]
    assert all("PAGE_TWO_ONLY" not in unit.text for unit in page_one_units)
    assert all("PAGE_ONE_ONLY" not in unit.text for unit in page_two_units)


def test_normalized_source_text_is_not_lost(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    source_text = (
        "First paragraph keeps its words.\n\n"
        "Second paragraph has    extra spacing but no lost content."
    )
    ingestion = ingest_synthetic_material(database_path, [source_text])

    units = generate_knowledge_units_for_material(
        database_path,
        ingestion.material.id,
    )

    reconstructed = " ".join(unit.text for unit in units)
    assert reconstructed == normalize_chunk_text(ingestion.pages[0].text)


def test_repeated_generation_replaces_without_duplicates(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    ingestion = ingest_synthetic_material(
        database_path,
        ["Stable page one.", "Stable page two."],
    )

    first = generate_knowledge_units_for_material(database_path, ingestion.material.id)
    second = generate_knowledge_units_for_material(database_path, ingestion.material.id)

    assert knowledge_unit_count(database_path) == len(second)
    assert [(unit.page_number, unit.chunk_index, unit.text) for unit in first] == [
        (unit.page_number, unit.chunk_index, unit.text) for unit in second
    ]


def test_materials_do_not_receive_each_others_pages(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material_a = ingest_synthetic_material(
        database_path,
        ["Material A source page."],
        title="Material A",
    )
    material_b = ingest_pdf(
        database_path=database_path,
        pdf_bytes=make_pdf(["Material B source page."]),
        original_filename="material-b.pdf",
        title="Material B",
    )

    units_a = generate_knowledge_units_for_material(
        database_path,
        material_a.material.id,
    )
    units_b = generate_knowledge_units_for_material(
        database_path,
        material_b.material.id,
    )

    assert {unit.extracted_page_id for unit in units_a} == {
        material_a.pages[0].id,
    }
    assert {unit.extracted_page_id for unit in units_b} == {
        material_b.pages[0].id,
    }


def test_cross_material_page_relationship_is_rejected(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material_a = ingest_synthetic_material(
        database_path,
        ["Material A page."],
        title="Material A",
    )
    material_b = ingest_pdf(
        database_path=database_path,
        pdf_bytes=make_pdf(["Material B page."]),
        original_filename="material-b.pdf",
        title="Material B",
    )

    with db.connect(database_path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO knowledge_units (
                    material_id, extracted_page_id, page_number,
                    chunk_index, text
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    material_a.material.id,
                    material_b.pages[0].id,
                    material_b.pages[0].page_number,
                    0,
                    "Invalid provenance.",
                ),
            )


def test_failed_generation_rolls_back_previous_units(tmp_path, monkeypatch):
    database_path = tmp_path / "recall.sqlite3"
    ingestion = ingest_synthetic_material(database_path, ["Safe page text."])
    original_units = generate_knowledge_units_for_material(
        database_path,
        ingestion.material.id,
    )

    def fail_after_delete(connection, *, material_id, units):
        connection.execute(
            "DELETE FROM knowledge_units WHERE material_id = ?",
            (material_id,),
        )
        raise RuntimeError("simulated insert failure")

    monkeypatch.setattr(db, "replace_knowledge_units_for_material", fail_after_delete)

    with pytest.raises(RuntimeError, match="simulated insert failure"):
        generate_knowledge_units_for_material(database_path, ingestion.material.id)

    rows = list_knowledge_units_for_material(database_path, ingestion.material.id)
    assert [(row["page_number"], row["chunk_index"], row["text"]) for row in rows] == [
        (unit.page_number, unit.chunk_index, unit.text) for unit in original_units
    ]


def test_missing_material_fails_without_units(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    initialize_database(database_path)

    with pytest.raises(KnowledgeUnitGenerationError, match="does not exist"):
        generate_knowledge_units_for_material(database_path, 999)

    assert knowledge_unit_count(database_path) == 0


def test_schema_version_is_upgraded_for_knowledge_units(tmp_path):
    database_path = tmp_path / "recall.sqlite3"

    initialize_database(database_path)

    assert current_schema_version(database_path) == 4
