from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from recall.answering import answer_question
from recall import db
from recall.config import Settings
from recall.evaluation.fixtures import FixtureError, load_fixture
from recall.evaluation.runner import run_evaluation, write_json_report
from recall.retrieval.embeddings import EmbeddingProviderError, build_embedding_provider


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Recall trust evaluation fixture."
    )
    parser.add_argument("fixture", type=Path, help="Path to evaluation fixture JSON.")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a structured JSON report.",
    )
    args = parser.parse_args()

    try:
        fixture = load_fixture(args.fixture)
        settings = Settings.from_env()
        validate_evaluation_state(settings.database_path)
        embedding_provider = build_embedding_provider(settings)
        results, summary = run_evaluation(
            fixture,
            lambda question: answer_question(
                database_path=settings.database_path,
                question=question,
                embedding_provider=embedding_provider,
            ),
        )
    except (FixtureError, EmbeddingProviderError, OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}")
        return 1

    print_summary(fixture.name, summary)
    print_case_lines(results)
    if args.json_output is not None:
        write_json_report(
            path=args.json_output,
            fixture=fixture,
            results=results,
            summary=summary,
        )
        print(f"JSON report written: {args.json_output}")
    return 0


def validate_evaluation_state(database_path: Path) -> None:
    if not database_path.is_file():
        raise ValueError(f"Evaluation database does not exist: {database_path}")
    try:
        with db.connect(database_path) as connection:
            knowledge_units = connection.execute(
                "SELECT COUNT(*) AS count FROM knowledge_units"
            ).fetchone()["count"]
            embeddings = connection.execute(
                "SELECT COUNT(*) AS count FROM knowledge_unit_embeddings"
            ).fetchone()["count"]
    except sqlite3.Error as exc:
        raise ValueError(
            "Evaluation database is not initialized for Recall retrieval."
        ) from exc
    if knowledge_units < 1 or embeddings < 1:
        raise ValueError(
            "Evaluation database must contain KnowledgeUnits and embeddings."
        )


def print_summary(name, summary) -> None:
    print(f"Evaluation: {name}")
    print(f"Total cases: {summary.total_cases}")
    print(f"Status distribution: {summary.status_distribution}")
    print(f"Expected status matches: {summary.expected_status_matches}")
    print(f"Expected status mismatches: {summary.expected_status_mismatches}")
    print(f"False supported: {summary.false_supported_count}")
    print(f"False refusals: {summary.false_refusal_count}")
    print(
        "Answer-bearing without citation: "
        f"{summary.answer_bearing_without_citation_count}"
    )
    print(f"NOT_FOUND with citation: {summary.not_found_with_citation_count}")
    print(f"Invalid provenance: {summary.provenance_invalid_count}")
    print(f"Missing page numbers: {summary.page_missing_count}")
    print(f"Human review: {summary.human_review_distribution}")


def print_case_lines(results) -> None:
    for result in results:
        violations = ",".join(item.value for item in result.trust_violations) or "none"
        print(
            f"{result.case_id}: expected={result.expected_status.value} "
            f"actual={result.actual_status.value} "
            f"citations={result.citation_count} "
            f"human={result.human_review.value} "
            f"violations={violations}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
