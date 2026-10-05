from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ExpectedSupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    NOT_FOUND = "NOT_FOUND"


class HumanReview(str, Enum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    UNREVIEWED = "UNREVIEWED"


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_status: ExpectedSupportStatus
    reference_notes: str | None = None
    expected_concepts: list[str] = field(default_factory=list)
    expected_pages: list[int] = field(default_factory=list)
    human_review: HumanReview = HumanReview.UNREVIEWED


@dataclass(frozen=True)
class EvaluationFixture:
    name: str
    cases: list[EvaluationCase]


class FixtureError(ValueError):
    """Raised when an evaluation fixture is invalid."""


def load_fixture(path: Path) -> EvaluationFixture:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FixtureError(f"Could not read evaluation fixture: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FixtureError(f"Evaluation fixture is not valid JSON: {path}") from exc
    return parse_fixture(data)


def parse_fixture(data: Any) -> EvaluationFixture:
    if not isinstance(data, dict):
        raise FixtureError("Evaluation fixture must be a JSON object.")

    name = data.get("name")
    raw_cases = data.get("cases")
    if not isinstance(name, str) or not name.strip():
        raise FixtureError("Evaluation fixture requires a non-empty name.")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise FixtureError("Evaluation fixture requires a non-empty cases list.")

    cases = [parse_case(raw_case) for raw_case in raw_cases]
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise FixtureError("Evaluation case IDs must be unique.")

    return EvaluationFixture(name=name.strip(), cases=cases)


def parse_case(data: Any) -> EvaluationCase:
    if not isinstance(data, dict):
        raise FixtureError("Each evaluation case must be an object.")

    case_id = required_string(data, "case_id")
    question = required_string(data, "question")
    expected_status = parse_enum(
        ExpectedSupportStatus,
        required_string(data, "expected_status"),
        "expected_status",
    )
    reference_notes = optional_string(data, "reference_notes")
    expected_concepts = optional_string_list(data, "expected_concepts")
    expected_pages = optional_int_list(data, "expected_pages")
    human_review = parse_enum(
        HumanReview,
        data.get("human_review", HumanReview.UNREVIEWED.value),
        "human_review",
    )

    return EvaluationCase(
        case_id=case_id,
        question=question,
        expected_status=expected_status,
        reference_notes=reference_notes,
        expected_concepts=expected_concepts,
        expected_pages=expected_pages,
        human_review=human_review,
    )


def required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FixtureError(f"Evaluation case requires a non-empty {key}.")
    return value.strip()


def optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise FixtureError(f"{key} must be a string when provided.")
    stripped = value.strip()
    return stripped or None


def optional_string_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise FixtureError(f"{key} must be a list of strings when provided.")
    return [item.strip() for item in value if item.strip()]


def optional_int_list(data: dict[str, Any], key: str) -> list[int]:
    value = data.get(key, [])
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise FixtureError(f"{key} must be a list of integers when provided.")
    return value


def parse_enum(enum_type: type[Enum], value: Any, field_name: str):
    if not isinstance(value, str):
        raise FixtureError(f"{field_name} must be a string.")
    try:
        return enum_type(value.strip())
    except ValueError:
        allowed = ", ".join(item.value for item in enum_type)
        raise FixtureError(f"{field_name} must be one of: {allowed}.") from None
