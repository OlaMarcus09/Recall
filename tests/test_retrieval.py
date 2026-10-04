from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pytest

from recall import db
from recall.db import (
    embedding_count,
    initialize_database,
    list_knowledge_units_for_material,
)
from recall.knowledge.service import generate_knowledge_units_for_material
from recall.retrieval.embeddings import EmbeddedText
from recall.retrieval.index import (
    IndexingError,
    index_material_embeddings,
)
from recall.retrieval.search import (
    RetrievalError,
    retrieve_relevant_knowledge_units,
)
from tests.test_knowledge_units import create_material_with_pages


@dataclass
class FakeEmbeddingProvider:
    model: str = "fake-model"
    provider: str = "fake"
    fail_after: int | None = None
    calls: int = 0

    def embed(self, text: str) -> EmbeddedText:
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated provider failure")
        lowered = text.lower()
        vector = [
            float(lowered.count("apple")),
            float(lowered.count("banana")),
            float(lowered.count("finance")),
        ]
        if text.strip().lower() == "zero":
            vector = [0.0, 0.0, 0.0]
        return EmbeddedText(vector=vector, provider=self.provider, model=self.model)


def material_with_units(database_path):
    material = create_material_with_pages(
        database_path,
        [
            "Apple orchard apple harvest.",
            "Banana market banana supply.",
            "Finance lecture capital budget.",
        ],
        title="Retrieval Fixture",
    )
    generate_knowledge_units_for_material(database_path, material["id"])
    return material


def test_indexes_embeddings_with_provider_identity_and_dimension(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    provider = FakeEmbeddingProvider()

    result = index_material_embeddings(database_path, material["id"], provider)

    assert result.provider == "fake"
    assert result.model == "fake-model"
    assert result.vector_dimension == 3
    assert result.knowledge_unit_count == 3
    assert embedding_count(database_path) == 3

    with db.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT knowledge_unit_id, provider, model, vector_dimension
            FROM knowledge_unit_embeddings
            ORDER BY knowledge_unit_id
            """
        ).fetchall()
    assert [row["provider"] for row in rows] == ["fake", "fake", "fake"]
    assert [row["model"] for row in rows] == ["fake-model"] * 3
    assert [row["vector_dimension"] for row in rows] == [3, 3, 3]


def test_indexing_is_idempotent_for_same_provider_model(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    provider = FakeEmbeddingProvider()

    first = index_material_embeddings(database_path, material["id"], provider)
    second = index_material_embeddings(database_path, material["id"], provider)

    assert first.knowledge_unit_count == second.knowledge_unit_count
    assert embedding_count(database_path) == 3


def test_model_change_keeps_embeddings_separate(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)

    index_material_embeddings(database_path, material["id"], FakeEmbeddingProvider())
    index_material_embeddings(
        database_path,
        material["id"],
        FakeEmbeddingProvider(model="fake-model-v2"),
    )

    assert embedding_count(database_path) == 6
    old_results = retrieve_relevant_knowledge_units(
        database_path,
        question="apple",
        provider=FakeEmbeddingProvider(),
        material_id=material["id"],
    )
    new_results = retrieve_relevant_knowledge_units(
        database_path,
        question="apple",
        provider=FakeEmbeddingProvider(model="fake-model-v2"),
        material_id=material["id"],
    )
    assert old_results[0].knowledge_unit_id == new_results[0].knowledge_unit_id


def test_stale_embeddings_after_knowledge_unit_change_do_not_participate(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    provider = FakeEmbeddingProvider()
    index_material_embeddings(database_path, material["id"], provider)

    first_unit = list_knowledge_units_for_material(database_path, material["id"])[0]
    with db.connect(database_path) as connection:
        connection.execute(
            "UPDATE knowledge_units SET text = ? WHERE id = ?",
            ("Changed text without reindexing.", first_unit["id"]),
        )

    with pytest.raises(RetrievalError, match="stale or incomplete"):
        retrieve_relevant_knowledge_units(
            database_path,
            question="apple",
            provider=FakeEmbeddingProvider(),
            material_id=material["id"],
        )


def test_partial_provider_failure_preserves_previous_complete_index(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    provider = FakeEmbeddingProvider()
    index_material_embeddings(database_path, material["id"], provider)

    with pytest.raises(RuntimeError, match="simulated provider failure"):
        index_material_embeddings(
            database_path,
            material["id"],
            FakeEmbeddingProvider(fail_after=1),
        )

    assert embedding_count(database_path) == 3
    results = retrieve_relevant_knowledge_units(
        database_path,
        question="banana",
        provider=FakeEmbeddingProvider(),
        material_id=material["id"],
    )
    assert "Banana" in results[0].text


def test_retrieval_result_shape_ranking_and_provenance(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    index_material_embeddings(database_path, material["id"], FakeEmbeddingProvider())

    results = retrieve_relevant_knowledge_units(
        database_path,
        question="apple",
        provider=FakeEmbeddingProvider(),
        top_k=2,
        material_id=material["id"],
    )

    assert len(results) == 2
    assert results[0].rank == 1
    assert results[0].similarity > results[1].similarity
    assert results[0].material_id == material["id"]
    assert results[0].material_title == "Retrieval Fixture"
    assert results[0].original_filename.endswith(".pdf")
    assert results[0].extracted_page_id > 0
    assert results[0].page_number == 1
    assert results[0].chunk_index == 0
    assert "Apple" in results[0].text


def test_tie_ordering_is_deterministic(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = create_material_with_pages(
        database_path,
        ["Apple tie first.", "Apple tie second."],
        title="Tie Fixture",
    )
    generate_knowledge_units_for_material(database_path, material["id"])
    index_material_embeddings(database_path, material["id"], FakeEmbeddingProvider())

    results = retrieve_relevant_knowledge_units(
        database_path,
        question="apple",
        provider=FakeEmbeddingProvider(),
        material_id=material["id"],
    )

    assert [(result.page_number, result.chunk_index) for result in results] == [
        (1, 0),
        (2, 0),
    ]


@pytest.mark.parametrize("top_k", [0, -1, 21])
def test_top_k_validation(tmp_path, top_k):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    index_material_embeddings(database_path, material["id"], FakeEmbeddingProvider())

    with pytest.raises(RetrievalError, match="top_k"):
        retrieve_relevant_knowledge_units(
            database_path,
            question="apple",
            provider=FakeEmbeddingProvider(),
            top_k=top_k,
            material_id=material["id"],
        )


def test_empty_query_rejected(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    index_material_embeddings(database_path, material["id"], FakeEmbeddingProvider())

    with pytest.raises(RetrievalError, match="cannot be empty"):
        retrieve_relevant_knowledge_units(
            database_path,
            question="   ",
            provider=FakeEmbeddingProvider(),
            material_id=material["id"],
        )


def test_zero_vector_query_yields_no_trustworthy_results(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    index_material_embeddings(database_path, material["id"], FakeEmbeddingProvider())

    with pytest.raises(RetrievalError, match="No valid embeddings"):
        retrieve_relevant_knowledge_units(
            database_path,
            question="zero",
            provider=FakeEmbeddingProvider(),
            material_id=material["id"],
        )


def test_incompatible_dimension_model_index_rejected(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    index_material_embeddings(database_path, material["id"], FakeEmbeddingProvider())

    class TwoDimensionalProvider(FakeEmbeddingProvider):
        def embed(self, text: str) -> EmbeddedText:
            return EmbeddedText(vector=[1.0, 0.0], provider=self.provider, model=self.model)

    with pytest.raises(RetrievalError, match="dimension"):
        retrieve_relevant_knowledge_units(
            database_path,
            question="apple",
            provider=TwoDimensionalProvider(),
            material_id=material["id"],
        )


def test_malformed_persisted_vector_is_skipped(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    index_material_embeddings(database_path, material["id"], FakeEmbeddingProvider())
    rows = list_knowledge_units_for_material(database_path, material["id"])

    with db.connect(database_path) as connection:
        connection.execute(
            """
            UPDATE knowledge_unit_embeddings
            SET vector_json = ?
            WHERE knowledge_unit_id = ?
            """,
            ("not-json", rows[0]["id"]),
        )

    results = retrieve_relevant_knowledge_units(
        database_path,
        question="banana",
        provider=FakeEmbeddingProvider(),
        material_id=material["id"],
    )

    assert all(result.knowledge_unit_id != rows[0]["id"] for result in results)
    assert "Banana" in results[0].text


def test_material_scope_limits_results(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material_a = material_with_units(database_path)
    material_b = create_material_with_pages(
        database_path,
        ["Apple outside scope."],
        title="Other Material",
    )
    generate_knowledge_units_for_material(database_path, material_b["id"])
    provider = FakeEmbeddingProvider()
    index_material_embeddings(database_path, material_a["id"], provider)
    index_material_embeddings(database_path, material_b["id"], provider)

    results = retrieve_relevant_knowledge_units(
        database_path,
        question="apple",
        provider=FakeEmbeddingProvider(),
        material_id=material_b["id"],
    )

    assert {result.material_id for result in results} == {material_b["id"]}


def test_embedding_output_cannot_override_persisted_provenance(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    index_material_embeddings(database_path, material["id"], FakeEmbeddingProvider())

    result = retrieve_relevant_knowledge_units(
        database_path,
        question="finance",
        provider=FakeEmbeddingProvider(),
        material_id=material["id"],
    )[0]

    with db.connect(database_path) as connection:
        persisted = connection.execute(
            """
            SELECT ku.material_id, ku.extracted_page_id, ku.page_number,
                   ku.chunk_index, m.title, m.original_filename
            FROM knowledge_units AS ku
            JOIN materials AS m ON m.id = ku.material_id
            WHERE ku.id = ?
            """,
            (result.knowledge_unit_id,),
        ).fetchone()

    assert result.material_id == persisted["material_id"]
    assert result.extracted_page_id == persisted["extracted_page_id"]
    assert result.page_number == persisted["page_number"]
    assert result.chunk_index == persisted["chunk_index"]
    assert result.material_title == persisted["title"]
    assert result.original_filename == persisted["original_filename"]


def test_indexing_rejects_empty_vectors(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)

    class EmptyVectorProvider(FakeEmbeddingProvider):
        def embed(self, text: str) -> EmbeddedText:
            return EmbeddedText(vector=[], provider=self.provider, model=self.model)

    with pytest.raises(IndexingError, match="cannot be empty"):
        index_material_embeddings(database_path, material["id"], EmptyVectorProvider())

    assert embedding_count(database_path) == 0


def test_db_failure_during_embedding_replacement_rolls_back(tmp_path, monkeypatch):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    provider = FakeEmbeddingProvider()
    index_material_embeddings(database_path, material["id"], provider)

    def fail_replacement(connection, **kwargs):
        connection.execute(
            """
            DELETE FROM knowledge_unit_embeddings
            WHERE knowledge_unit_id IN (
                SELECT id FROM knowledge_units WHERE material_id = ?
            )
            """,
            (kwargs["material_id"],),
        )
        raise RuntimeError("simulated db failure")

    monkeypatch.setattr(db, "replace_embeddings_for_material", fail_replacement)

    with pytest.raises(RuntimeError, match="simulated db failure"):
        index_material_embeddings(database_path, material["id"], provider)

    assert embedding_count(database_path) == 3


def test_retrieval_uses_service_only_without_answer_generation(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material = material_with_units(database_path)
    provider = FakeEmbeddingProvider()
    index_material_embeddings(database_path, material["id"], provider)

    results = retrieve_relevant_knowledge_units(
        database_path,
        question="finance",
        provider=provider,
        material_id=material["id"],
    )

    assert results[0].text.startswith("Finance")
