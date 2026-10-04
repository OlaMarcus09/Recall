from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from recall import db
from recall.generation.providers import (
    EvidenceItem,
    GeneratedAnswer,
    GenerationProvider,
)
from recall.retrieval.search import RetrievalResult


REFUSAL_MESSAGE = "I couldn't find enough information in the material I have."
PARTIAL_PREFIX = "The available material only partially supports an answer."
MIN_EVIDENCE_SCORE = 0.55
CLOSE_SCORE_RATIO = 0.97
MAX_GENERATION_EVIDENCE = 5


class SupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    NOT_FOUND = "NOT_FOUND"


class EvidenceSupportAssessment(str, Enum):
    NO_KNOWN_GAPS = "NO_KNOWN_GAPS"
    KNOWN_PARTIAL = "KNOWN_PARTIAL"


@dataclass(frozen=True)
class Citation:
    knowledge_unit_id: int
    material_id: int
    material_title: str
    original_filename: str
    extracted_page_id: int
    page_number: int
    chunk_index: int


@dataclass(frozen=True)
class GroundedResponse:
    status: SupportStatus
    answer: str
    citations: list[Citation]


class GroundingError(RuntimeError):
    """Raised when a grounded response cannot be constructed safely."""


def generate_grounded_response(
    *,
    database_path: Path,
    question: str,
    retrieved: list[RetrievalResult],
    provider: GenerationProvider,
    support_assessment: EvidenceSupportAssessment = EvidenceSupportAssessment.NO_KNOWN_GAPS,
) -> GroundedResponse:
    selected = select_evidence(retrieved)
    if not selected:
        return not_found_response()

    evidence_items = [
        EvidenceItem(
            knowledge_unit_id=result.knowledge_unit_id,
            text=result.text,
        )
        for result in selected
    ]

    try:
        generated = provider.generate(question=question, evidence=evidence_items)
    except Exception:
        return not_found_response()

    if not is_well_formed_generation(generated):
        return not_found_response()

    cited_ids = list(dict.fromkeys(generated.cited_knowledge_unit_ids))
    citations = validate_citations(
        database_path=database_path,
        cited_knowledge_unit_ids=cited_ids,
        supplied_knowledge_unit_ids={item.knowledge_unit_id for item in evidence_items},
    )
    if not citations:
        return not_found_response()

    if support_assessment == EvidenceSupportAssessment.KNOWN_PARTIAL:
        return GroundedResponse(
            status=SupportStatus.PARTIAL,
            answer=f"{PARTIAL_PREFIX} {generated.answer_text.strip()}",
            citations=citations,
        )

    return GroundedResponse(
        status=SupportStatus.SUPPORTED,
        answer=generated.answer_text.strip(),
        citations=citations,
    )


def select_evidence(retrieved: list[RetrievalResult]) -> list[RetrievalResult]:
    ordered = sorted(retrieved, key=lambda result: result.rank)
    if not ordered:
        return []

    top_score = ordered[0].similarity
    if top_score < MIN_EVIDENCE_SCORE:
        return []

    close_score = max(MIN_EVIDENCE_SCORE, top_score * CLOSE_SCORE_RATIO)
    return [
        result
        for result in ordered[:MAX_GENERATION_EVIDENCE]
        if result.similarity >= close_score
    ]


def validate_citations(
    *,
    database_path: Path,
    cited_knowledge_unit_ids: list[int],
    supplied_knowledge_unit_ids: set[int],
) -> list[Citation]:
    if not cited_knowledge_unit_ids:
        return []
    if any(not isinstance(value, int) for value in cited_knowledge_unit_ids):
        return []

    cited_set = set(cited_knowledge_unit_ids)
    if not cited_set.issubset(supplied_knowledge_unit_ids):
        return []

    with db.connect(database_path) as connection:
        placeholders = ",".join("?" for _ in cited_knowledge_unit_ids)
        rows = connection.execute(
            f"""
            SELECT
                ku.id AS knowledge_unit_id,
                ku.material_id,
                ku.extracted_page_id,
                ku.page_number,
                ku.chunk_index,
                m.title AS material_title,
                m.original_filename
            FROM knowledge_units AS ku
            JOIN materials AS m ON m.id = ku.material_id
            WHERE ku.id IN ({placeholders})
            """,
            tuple(cited_knowledge_unit_ids),
        ).fetchall()

    if len(rows) != len(cited_set):
        return []

    citation_by_id = {
        row["knowledge_unit_id"]: Citation(
            knowledge_unit_id=row["knowledge_unit_id"],
            material_id=row["material_id"],
            material_title=row["material_title"],
            original_filename=row["original_filename"],
            extracted_page_id=row["extracted_page_id"],
            page_number=row["page_number"],
            chunk_index=row["chunk_index"],
        )
        for row in rows
    }
    return [
        citation_by_id[knowledge_unit_id]
        for knowledge_unit_id in cited_knowledge_unit_ids
    ]


def is_well_formed_generation(value: Any) -> bool:
    if not isinstance(value, GeneratedAnswer):
        return False
    if not value.answer_text or not value.answer_text.strip():
        return False
    if not isinstance(value.cited_knowledge_unit_ids, list):
        return False
    return all(isinstance(item, int) for item in value.cited_knowledge_unit_ids)


def not_found_response() -> GroundedResponse:
    return GroundedResponse(
        status=SupportStatus.NOT_FOUND,
        answer=REFUSAL_MESSAGE,
        citations=[],
    )
