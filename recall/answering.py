from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recall import db
from recall.confusion.service import record_question_memory
from recall.generation.providers import ExtractiveGenerationProvider, GenerationProvider
from recall.generation.service import (
    REFUSAL_MESSAGE,
    SupportStatus,
    generate_grounded_response,
)
from recall.retrieval.embeddings import EmbeddingProvider
from recall.retrieval.search import retrieve_relevant_knowledge_units


MAX_QUESTION_LENGTH = 500
CITATION_EXCERPT_CHARS = 280
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class LearnerCitation:
    knowledge_unit_id: int
    material_id: int
    document_title: str
    original_filename: str
    page_number: int
    chunk_index: int
    evidence_excerpt: str


@dataclass(frozen=True)
class LearnerAnswer:
    question: str
    status: SupportStatus
    answer: str
    citations: list[LearnerCitation]


class QuestionValidationError(ValueError):
    """Raised when learner question input is not acceptable for answering."""


def answer_question(
    *,
    database_path: Path,
    question: str,
    embedding_provider: EmbeddingProvider,
    generation_provider: GenerationProvider | None = None,
    top_k: int = DEFAULT_TOP_K,
    remember_question: bool = False,
) -> LearnerAnswer:
    normalized_question = normalize_question(question)
    provider = generation_provider or ExtractiveGenerationProvider()

    try:
        retrieved = retrieve_relevant_knowledge_units(
            database_path,
            question=normalized_question,
            provider=embedding_provider,
            top_k=top_k,
        )
        grounded = generate_grounded_response(
            database_path=database_path,
            question=normalized_question,
            retrieved=retrieved,
            provider=provider,
        )
    except Exception:
        grounded = generate_grounded_response(
            database_path=database_path,
            question=normalized_question,
            retrieved=[],
            provider=provider,
        )

    learner_answer = LearnerAnswer(
        question=normalized_question,
        status=grounded.status,
        answer=grounded.answer,
        citations=[
            LearnerCitation(
                knowledge_unit_id=citation.knowledge_unit_id,
                material_id=citation.material_id,
                document_title=citation.material_title,
                original_filename=citation.original_filename,
                page_number=citation.page_number,
                chunk_index=citation.chunk_index,
                evidence_excerpt=load_bounded_evidence_excerpt(
                    database_path,
                    citation.knowledge_unit_id,
                ),
            )
            for citation in grounded.citations
        ],
    )
    if remember_question:
        try:
            record_question_memory(
                database_path=database_path,
                question=normalized_question,
                support_status=grounded.status,
                embedding_provider=embedding_provider,
            )
        except Exception:
            pass
    return learner_answer


def normalize_question(question: str) -> str:
    normalized = question.strip()
    if not normalized:
        raise QuestionValidationError("Enter a question before asking Recall.")
    if len(normalized) > MAX_QUESTION_LENGTH:
        raise QuestionValidationError(
            f"Questions must be {MAX_QUESTION_LENGTH} characters or fewer."
        )
    return normalized


def load_bounded_evidence_excerpt(
    database_path: Path,
    knowledge_unit_id: int,
) -> str:
    with db.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT text
            FROM knowledge_units
            WHERE id = ?
            """,
            (knowledge_unit_id,),
        ).fetchone()
    if row is None:
        return ""
    return bounded_excerpt(row["text"], CITATION_EXCERPT_CHARS)


def bounded_excerpt(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    truncated = normalized[: max(0, limit - 1)].rstrip()
    return f"{truncated}..."
