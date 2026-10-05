from __future__ import annotations

from dataclasses import dataclass

import pytest

from recall import db
from recall.answering import (
    MAX_QUESTION_LENGTH,
    QuestionValidationError,
    answer_question,
    bounded_excerpt,
)
from recall.db import initialize_database
from recall.generation.providers import EvidenceItem, GeneratedAnswer
from recall.generation.service import REFUSAL_MESSAGE, SupportStatus
from recall.knowledge.service import generate_knowledge_units_for_material
from recall.retrieval.embeddings import EmbeddedText
from recall.retrieval.index import index_material_embeddings
from tests.test_knowledge_units import create_material_with_pages


@dataclass
class FakeEmbeddingProvider:
    provider: str = "fake"
    model: str = "fake-model"
    fail: bool = False

    def embed(self, text: str) -> EmbeddedText:
        if self.fail:
            raise RuntimeError("simulated embedding failure")
        lowered = text.lower()
        return EmbeddedText(
            vector=[
                float(lowered.count("strategy") + lowered.count("strategic")),
                float(lowered.count("mission")),
                float(lowered.count("vision")),
                1.0,
            ],
            provider=self.provider,
            model=self.model,
        )


@dataclass
class FakeGenerationProvider:
    response: GeneratedAnswer | None = None
    exc: Exception | None = None
    provider: str = "fake-generation"
    model: str = "fake-model"
    calls: int = 0
    last_question: str | None = None
    last_evidence: list[EvidenceItem] | None = None

    def generate(self, *, question: str, evidence: list[EvidenceItem]):
        self.calls += 1
        self.last_question = question
        self.last_evidence = evidence
        if self.exc is not None:
            raise self.exc
        return self.response


def indexed_material(database_path, page_texts):
    material = create_material_with_pages(
        database_path,
        page_texts,
        title="Synthetic Strategy Notes",
    )
    units = generate_knowledge_units_for_material(database_path, material["id"])
    index_material_embeddings(database_path, material["id"], FakeEmbeddingProvider())
    return material, units


def test_answer_question_returns_supported_with_db_citation_excerpt(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    material, units = indexed_material(
        database_path,
        ["Strategy creates sustainable advantage for an organisation."],
    )
    provider = FakeGenerationProvider(
        response=GeneratedAnswer(
            answer_text="Strategy creates sustainable advantage.",
            cited_knowledge_unit_ids=[units[0].id],
        )
    )

    result = answer_question(
        database_path=database_path,
        question="  What does strategy create?  ",
        embedding_provider=FakeEmbeddingProvider(),
        generation_provider=provider,
    )

    assert result.question == "What does strategy create?"
    assert result.status == SupportStatus.SUPPORTED
    assert result.answer == "Strategy creates sustainable advantage."
    assert provider.calls == 1
    assert result.citations[0].knowledge_unit_id == units[0].id
    assert result.citations[0].material_id == material["id"]
    assert result.citations[0].document_title == "Synthetic Strategy Notes"
    assert result.citations[0].page_number == 1
    assert result.citations[0].chunk_index == 0
    assert "sustainable advantage" in result.citations[0].evidence_excerpt


def test_answer_question_can_return_multiple_citations(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    _, units = indexed_material(
        database_path,
        [
            "Mission describes why the organisation exists.",
            "Vision describes the desired future direction.",
        ],
    )
    provider = FakeGenerationProvider(
        response=GeneratedAnswer(
            answer_text="Mission explains purpose; vision explains future direction.",
            cited_knowledge_unit_ids=[units[0].id, units[1].id],
        )
    )

    result = answer_question(
        database_path=database_path,
        question="How are mission and vision different?",
        embedding_provider=FakeEmbeddingProvider(),
        generation_provider=provider,
    )

    assert result.status == SupportStatus.SUPPORTED
    assert [citation.knowledge_unit_id for citation in result.citations] == [
        units[0].id,
        units[1].id,
    ]


def test_answer_question_handles_missing_index_as_not_found(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    initialize_database(database_path)
    provider = FakeGenerationProvider(
        response=GeneratedAnswer(answer_text="Should not be used.", cited_knowledge_unit_ids=[])
    )

    result = answer_question(
        database_path=database_path,
        question="What does strategy create?",
        embedding_provider=FakeEmbeddingProvider(),
        generation_provider=provider,
    )

    assert result.status == SupportStatus.NOT_FOUND
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []
    assert provider.calls == 0


def test_answer_question_handles_provider_failure_safely(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    indexed_material(
        database_path,
        ["Strategy creates sustainable advantage for an organisation."],
    )

    result = answer_question(
        database_path=database_path,
        question="What does strategy create?",
        embedding_provider=FakeEmbeddingProvider(fail=True),
        generation_provider=FakeGenerationProvider(
            response=GeneratedAnswer(answer_text="Nope.", cited_knowledge_unit_ids=[])
        ),
    )

    assert result.status == SupportStatus.NOT_FOUND
    assert result.answer == REFUSAL_MESSAGE
    assert result.citations == []


def test_answer_question_rejects_blank_and_overlong_questions(tmp_path):
    with pytest.raises(QuestionValidationError, match="Enter a question"):
        answer_question(
            database_path=tmp_path / "recall.sqlite3",
            question="   ",
            embedding_provider=FakeEmbeddingProvider(),
        )

    with pytest.raises(QuestionValidationError, match="500 characters"):
        answer_question(
            database_path=tmp_path / "recall.sqlite3",
            question="x" * (MAX_QUESTION_LENGTH + 1),
            embedding_provider=FakeEmbeddingProvider(),
        )


def test_citation_excerpt_is_bounded_and_from_persisted_text(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    _, units = indexed_material(
        database_path,
        ["Strategy " + ("evidence " * 35)],
    )
    provider = FakeGenerationProvider(
        response=GeneratedAnswer(
            answer_text="The answer cites the stored unit.",
            cited_knowledge_unit_ids=[units[0].id],
        )
    )

    result = answer_question(
        database_path=database_path,
        question="strategy",
        embedding_provider=FakeEmbeddingProvider(),
        generation_provider=provider,
    )

    excerpt = result.citations[0].evidence_excerpt
    assert len(excerpt) <= 280 + 3
    assert excerpt.startswith("Strategy evidence")
    assert excerpt.endswith("...")

    with db.connect(database_path) as connection:
        stored = connection.execute(
            "SELECT text FROM knowledge_units WHERE id = ?",
            (units[0].id,),
        ).fetchone()["text"]
    assert bounded_excerpt(stored, 280) == excerpt
