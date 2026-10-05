from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from recall.answering import LearnerAnswer
from recall.evaluation.fixtures import (
    EvaluationCase,
    EvaluationFixture,
    ExpectedSupportStatus,
    HumanReview,
)
from recall.generation.service import SupportStatus


class TrustViolation(str, Enum):
    MISSING_CITATION = "MISSING_CITATION"
    UNEXPECTED_CITATION = "UNEXPECTED_CITATION"
    MISSING_PROVENANCE = "MISSING_PROVENANCE"
    MISSING_PAGE_NUMBER = "MISSING_PAGE_NUMBER"
    FALSE_SUPPORTED = "FALSE_SUPPORTED"
    FALSE_REFUSAL = "FALSE_REFUSAL"


@dataclass(frozen=True)
class CitationEvaluation:
    knowledge_unit_id: int
    material_id: int
    document_title: str
    page_number: int | None
    chunk_index: int
    provenance_valid: bool


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    question: str
    expected_status: ExpectedSupportStatus
    actual_status: SupportStatus
    status_match: bool
    answer: str
    citation_count: int
    citations: list[CitationEvaluation]
    trust_violations: list[TrustViolation]
    human_review: HumanReview
    reference_notes: str | None
    expected_concepts: list[str]
    expected_pages: list[int]


@dataclass(frozen=True)
class EvaluationSummary:
    total_cases: int
    status_distribution: dict[str, int]
    expected_status_matches: int
    expected_status_mismatches: int
    false_supported_count: int
    false_refusal_count: int
    answer_bearing_without_citation_count: int
    not_found_with_citation_count: int
    provenance_invalid_count: int
    page_missing_count: int
    unsupported_refusal_matches: int
    human_review_distribution: dict[str, int]


AnswerFunction = Callable[[str], LearnerAnswer]


def run_evaluation(
    fixture: EvaluationFixture,
    answer_function: AnswerFunction,
) -> tuple[list[EvaluationResult], EvaluationSummary]:
    results = [evaluate_case(case, answer_function) for case in fixture.cases]
    return results, summarize_results(results)


def evaluate_case(
    case: EvaluationCase,
    answer_function: AnswerFunction,
) -> EvaluationResult:
    answer = answer_function(case.question)
    citations = [
        CitationEvaluation(
            knowledge_unit_id=citation.knowledge_unit_id,
            material_id=citation.material_id,
            document_title=citation.document_title,
            page_number=citation.page_number,
            chunk_index=citation.chunk_index,
            provenance_valid=has_valid_provenance(citation),
        )
        for citation in answer.citations
    ]
    violations = detect_violations(case, answer, citations)
    expected_status = SupportStatus(case.expected_status.value)

    return EvaluationResult(
        case_id=case.case_id,
        question=case.question,
        expected_status=case.expected_status,
        actual_status=answer.status,
        status_match=answer.status == expected_status,
        answer=answer.answer,
        citation_count=len(answer.citations),
        citations=citations,
        trust_violations=violations,
        human_review=case.human_review,
        reference_notes=case.reference_notes,
        expected_concepts=list(case.expected_concepts),
        expected_pages=list(case.expected_pages),
    )


def has_valid_provenance(citation: Any) -> bool:
    return (
        isinstance(citation.knowledge_unit_id, int)
        and citation.knowledge_unit_id > 0
        and isinstance(citation.material_id, int)
        and citation.material_id > 0
        and bool(citation.document_title)
        and isinstance(citation.chunk_index, int)
        and citation.chunk_index >= 0
    )


def detect_violations(
    case: EvaluationCase,
    answer: LearnerAnswer,
    citations: list[CitationEvaluation],
) -> list[TrustViolation]:
    violations: list[TrustViolation] = []
    answer_bearing = answer.status in {SupportStatus.SUPPORTED, SupportStatus.PARTIAL}

    if answer_bearing and not citations:
        violations.append(TrustViolation.MISSING_CITATION)
    if answer.status == SupportStatus.NOT_FOUND and citations:
        violations.append(TrustViolation.UNEXPECTED_CITATION)
    if case.expected_status == ExpectedSupportStatus.NOT_FOUND and answer_bearing:
        violations.append(TrustViolation.FALSE_SUPPORTED)
    if (
        case.expected_status
        in {ExpectedSupportStatus.SUPPORTED, ExpectedSupportStatus.PARTIAL}
        and answer.status == SupportStatus.NOT_FOUND
    ):
        violations.append(TrustViolation.FALSE_REFUSAL)
    if any(not citation.provenance_valid for citation in citations):
        violations.append(TrustViolation.MISSING_PROVENANCE)
    if any(citation.page_number is None or citation.page_number < 1 for citation in citations):
        violations.append(TrustViolation.MISSING_PAGE_NUMBER)

    return violations


def summarize_results(results: list[EvaluationResult]) -> EvaluationSummary:
    status_distribution = Counter(result.actual_status.value for result in results)
    human_distribution = Counter(result.human_review.value for result in results)
    violations = Counter(
        violation.value
        for result in results
        for violation in result.trust_violations
    )

    return EvaluationSummary(
        total_cases=len(results),
        status_distribution=dict(status_distribution),
        expected_status_matches=sum(1 for result in results if result.status_match),
        expected_status_mismatches=sum(1 for result in results if not result.status_match),
        false_supported_count=violations[TrustViolation.FALSE_SUPPORTED.value],
        false_refusal_count=violations[TrustViolation.FALSE_REFUSAL.value],
        answer_bearing_without_citation_count=violations[
            TrustViolation.MISSING_CITATION.value
        ],
        not_found_with_citation_count=violations[
            TrustViolation.UNEXPECTED_CITATION.value
        ],
        provenance_invalid_count=violations[
            TrustViolation.MISSING_PROVENANCE.value
        ],
        page_missing_count=violations[TrustViolation.MISSING_PAGE_NUMBER.value],
        unsupported_refusal_matches=sum(
            1
            for result in results
            if result.expected_status == ExpectedSupportStatus.NOT_FOUND
            and result.actual_status == SupportStatus.NOT_FOUND
        ),
        human_review_distribution=dict(human_distribution),
    )


def report_to_dict(
    *,
    fixture: EvaluationFixture,
    results: list[EvaluationResult],
    summary: EvaluationSummary,
) -> dict[str, Any]:
    return {
        "fixture": fixture.name,
        "summary": serialize(summary),
        "results": [serialize(result) for result in results],
    }


def write_json_report(
    *,
    path: Path,
    fixture: EvaluationFixture,
    results: list[EvaluationResult],
    summary: EvaluationSummary,
) -> None:
    path.write_text(
        json.dumps(
            report_to_dict(fixture=fixture, results=results, summary=summary),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return serialize(asdict(value))
    return value
