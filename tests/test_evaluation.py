from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from recall.answering import LearnerAnswer, LearnerCitation
from recall.evaluation.__main__ import main as evaluation_main
from recall.evaluation.fixtures import (
    EvaluationCase,
    EvaluationFixture,
    ExpectedSupportStatus,
    FixtureError,
    HumanReview,
    load_fixture,
    parse_fixture,
)
from recall.evaluation.runner import (
    TrustViolation,
    report_to_dict,
    run_evaluation,
    write_json_report,
)
from recall.generation.service import REFUSAL_MESSAGE, SupportStatus
from recall.retrieval.embeddings import EmbeddedText, EmbeddingProviderError


FIXTURE_PATH = Path("tests/fixtures/evaluation/synthetic_trust_fixture.json")


@dataclass
class FakeEmbeddingProvider:
    provider: str = "fake"
    model: str = "fake-model"

    def embed(self, text: str) -> EmbeddedText:
        return EmbeddedText(vector=[1.0], provider=self.provider, model=self.model)


def learner_citation(**overrides):
    values = {
        "knowledge_unit_id": 10,
        "material_id": 3,
        "document_title": "Synthetic Notes",
        "original_filename": "synthetic.pdf",
        "page_number": 2,
        "chunk_index": 0,
        "evidence_excerpt": "Synthetic evidence.",
    }
    values.update(overrides)
    return LearnerCitation(**values)


def learner_answer(
    status: SupportStatus,
    *,
    citations: list[LearnerCitation] | None = None,
    answer: str = "Synthetic answer.",
) -> LearnerAnswer:
    return LearnerAnswer(
        question="runtime question",
        status=status,
        answer=answer,
        citations=citations or [],
    )


def fixture_with_cases(*cases: EvaluationCase) -> EvaluationFixture:
    return EvaluationFixture(name="unit fixture", cases=list(cases))


def case(
    case_id: str,
    expected_status: ExpectedSupportStatus,
    question: str = "Runtime question?",
    **kwargs,
) -> EvaluationCase:
    return EvaluationCase(
        case_id=case_id,
        question=question,
        expected_status=expected_status,
        **kwargs,
    )


def test_evaluation_fixture_parsing_and_stable_case_ids():
    fixture = load_fixture(FIXTURE_PATH)

    assert fixture.name == "synthetic trust fixture"
    assert [item.case_id for item in fixture.cases] == ["SYN-01", "SYN-02", "SYN-03"]
    assert fixture.cases[0].expected_status == ExpectedSupportStatus.SUPPORTED
    assert fixture.cases[0].expected_concepts == ["strategy provides direction"]
    assert fixture.cases[0].human_review == HumanReview.UNREVIEWED
    assert fixture.cases[1].human_review == HumanReview.CORRECT


@pytest.mark.parametrize(
    "payload,error",
    [
        ({}, "non-empty name"),
        ({"name": "bad", "cases": []}, "non-empty cases"),
        (
            {
                "name": "bad",
                "cases": [
                    {
                        "case_id": "A",
                        "question": "Question?",
                        "expected_status": "BOGUS",
                    }
                ],
            },
            "expected_status",
        ),
        (
            {
                "name": "bad",
                "cases": [
                    {
                        "case_id": "A",
                        "question": "Question?",
                        "expected_status": "SUPPORTED",
                    },
                    {
                        "case_id": "A",
                        "question": "Another?",
                        "expected_status": "SUPPORTED",
                    },
                ],
            },
            "unique",
        ),
    ],
)
def test_invalid_fixture_rejection(payload, error):
    with pytest.raises(FixtureError, match=error):
        parse_fixture(payload)


def test_expected_metadata_never_enters_runtime_question():
    recorded_questions: list[str] = []
    fixture = fixture_with_cases(
        case(
            "META-01",
            ExpectedSupportStatus.SUPPORTED,
            question="Only this text reaches runtime.",
            expected_concepts=["do not leak this concept"],
            reference_notes="do not leak this note",
        )
    )

    def answer_function(question):
        recorded_questions.append(question)
        return learner_answer(SupportStatus.SUPPORTED, citations=[learner_citation()])

    results, _ = run_evaluation(fixture, answer_function)

    assert recorded_questions == ["Only this text reaches runtime."]
    assert results[0].expected_concepts == ["do not leak this concept"]
    assert results[0].reference_notes == "do not leak this note"


def test_supported_result_with_citation_records_provenance():
    fixture = fixture_with_cases(case("SUP-01", ExpectedSupportStatus.SUPPORTED))

    results, summary = run_evaluation(
        fixture,
        lambda question: learner_answer(
            SupportStatus.SUPPORTED,
            citations=[learner_citation()],
        ),
    )

    assert results[0].status_match is True
    assert results[0].citation_count == 1
    assert results[0].citations[0].document_title == "Synthetic Notes"
    assert results[0].citations[0].page_number == 2
    assert results[0].citations[0].provenance_valid is True
    assert results[0].trust_violations == []
    assert summary.expected_status_matches == 1


def test_not_found_with_zero_citations_records_refusal():
    fixture = fixture_with_cases(case("NF-01", ExpectedSupportStatus.NOT_FOUND))

    results, summary = run_evaluation(
        fixture,
        lambda question: learner_answer(
            SupportStatus.NOT_FOUND,
            answer=REFUSAL_MESSAGE,
        ),
    )

    assert results[0].status_match is True
    assert results[0].citation_count == 0
    assert results[0].trust_violations == []
    assert summary.unsupported_refusal_matches == 1


def test_expected_status_mismatch_is_reported():
    fixture = fixture_with_cases(case("MISMATCH-01", ExpectedSupportStatus.SUPPORTED))

    results, summary = run_evaluation(
        fixture,
        lambda question: learner_answer(SupportStatus.PARTIAL, citations=[learner_citation()]),
    )

    assert results[0].status_match is False
    assert summary.expected_status_mismatches == 1


def test_false_supported_detection_for_expected_not_found():
    fixture = fixture_with_cases(case("UNSUPPORTED-01", ExpectedSupportStatus.NOT_FOUND))

    results, summary = run_evaluation(
        fixture,
        lambda question: learner_answer(SupportStatus.SUPPORTED, citations=[learner_citation()]),
    )

    assert TrustViolation.FALSE_SUPPORTED in results[0].trust_violations
    assert summary.false_supported_count == 1


def test_false_refusal_detection_for_expected_supported():
    fixture = fixture_with_cases(case("REFUSAL-01", ExpectedSupportStatus.SUPPORTED))

    results, summary = run_evaluation(
        fixture,
        lambda question: learner_answer(SupportStatus.NOT_FOUND, answer=REFUSAL_MESSAGE),
    )

    assert TrustViolation.FALSE_REFUSAL in results[0].trust_violations
    assert summary.false_refusal_count == 1


def test_missing_citation_detection_for_answer_bearing_state():
    fixture = fixture_with_cases(case("NO-CITE-01", ExpectedSupportStatus.SUPPORTED))

    results, summary = run_evaluation(
        fixture,
        lambda question: learner_answer(SupportStatus.SUPPORTED),
    )

    assert TrustViolation.MISSING_CITATION in results[0].trust_violations
    assert summary.answer_bearing_without_citation_count == 1


def test_unexpected_citation_detection_for_not_found():
    fixture = fixture_with_cases(case("BAD-NF-01", ExpectedSupportStatus.NOT_FOUND))

    results, summary = run_evaluation(
        fixture,
        lambda question: learner_answer(
            SupportStatus.NOT_FOUND,
            citations=[learner_citation()],
            answer=REFUSAL_MESSAGE,
        ),
    )

    assert TrustViolation.UNEXPECTED_CITATION in results[0].trust_violations
    assert summary.not_found_with_citation_count == 1


def test_provenance_and_page_number_validation():
    fixture = fixture_with_cases(case("BAD-PROV-01", ExpectedSupportStatus.SUPPORTED))

    results, summary = run_evaluation(
        fixture,
        lambda question: learner_answer(
            SupportStatus.SUPPORTED,
            citations=[
                learner_citation(
                    knowledge_unit_id=0,
                    material_id=0,
                    document_title="",
                    page_number=0,
                )
            ],
        ),
    )

    assert TrustViolation.MISSING_PROVENANCE in results[0].trust_violations
    assert TrustViolation.MISSING_PAGE_NUMBER in results[0].trust_violations
    assert summary.provenance_invalid_count == 1
    assert summary.page_missing_count == 1


def test_human_review_defaults_to_unreviewed_and_semantic_correctness_not_fabricated():
    fixture = fixture_with_cases(case("HUMAN-01", ExpectedSupportStatus.SUPPORTED))

    results, summary = run_evaluation(
        fixture,
        lambda question: learner_answer(SupportStatus.SUPPORTED, citations=[learner_citation()]),
    )

    assert results[0].human_review == HumanReview.UNREVIEWED
    assert summary.human_review_distribution == {"UNREVIEWED": 1}
    assert not hasattr(results[0], "semantic_correct")


def test_summary_counts_multiple_metrics():
    fixture = fixture_with_cases(
        case("OK-01", ExpectedSupportStatus.SUPPORTED, question="ok question"),
        case("MISS-01", ExpectedSupportStatus.SUPPORTED, question="miss question"),
        case("BAD-01", ExpectedSupportStatus.NOT_FOUND, question="bad question"),
    )
    answers = {
        "OK-01": learner_answer(SupportStatus.SUPPORTED, citations=[learner_citation()]),
        "MISS-01": learner_answer(SupportStatus.NOT_FOUND, answer=REFUSAL_MESSAGE),
        "BAD-01": learner_answer(SupportStatus.PARTIAL, citations=[learner_citation()]),
    }

    results, summary = run_evaluation(
        fixture,
        lambda question: answers[
            next(item.case_id for item in fixture.cases if item.question == question)
        ],
    )

    assert len(results) == 3
    assert summary.total_cases == 3
    assert summary.status_distribution == {
        "SUPPORTED": 1,
        "NOT_FOUND": 1,
        "PARTIAL": 1,
    }
    assert summary.expected_status_matches == 1
    assert summary.false_refusal_count == 1
    assert summary.false_supported_count == 1


def test_json_report_serialization(tmp_path):
    fixture = fixture_with_cases(case("JSON-01", ExpectedSupportStatus.SUPPORTED))
    results, summary = run_evaluation(
        fixture,
        lambda question: learner_answer(SupportStatus.SUPPORTED, citations=[learner_citation()]),
    )
    output_path = tmp_path / "report.json"

    write_json_report(
        path=output_path,
        fixture=fixture,
        results=results,
        summary=summary,
    )

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == report_to_dict(fixture=fixture, results=results, summary=summary)
    assert written["results"][0]["actual_status"] == "SUPPORTED"


def test_cli_success_on_deterministic_synthetic_fixture(monkeypatch, capsys, tmp_path):
    def fake_build_provider(settings):
        return FakeEmbeddingProvider()

    def fake_answer_question(**kwargs):
        question = kwargs["question"]
        if "dosage" in question:
            return learner_answer(SupportStatus.NOT_FOUND, answer=REFUSAL_MESSAGE)
        return learner_answer(SupportStatus.SUPPORTED, citations=[learner_citation()])

    output_path = tmp_path / "report.json"
    monkeypatch.setattr(
        "recall.evaluation.__main__.build_embedding_provider",
        fake_build_provider,
    )
    monkeypatch.setattr(
        "recall.evaluation.__main__.answer_question",
        fake_answer_question,
    )
    monkeypatch.setattr(
        "recall.evaluation.__main__.validate_evaluation_state",
        lambda database_path: None,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "recall.evaluation",
            str(FIXTURE_PATH),
            "--json-output",
            str(output_path),
        ],
    )

    assert evaluation_main() == 0

    captured = capsys.readouterr()
    assert "Evaluation: synthetic trust fixture" in captured.out
    assert "Total cases: 3" in captured.out
    assert output_path.exists()


def test_cli_clear_failure_for_unavailable_database(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("RECALL_DATABASE_PATH", str(tmp_path / "missing.sqlite3"))
    monkeypatch.setattr("sys.argv", ["recall.evaluation", str(FIXTURE_PATH)])

    assert evaluation_main() == 1

    captured = capsys.readouterr()
    assert "Evaluation database does not exist" in captured.out


def test_cli_clear_failure_for_unavailable_provider(monkeypatch, capsys):
    def fake_build_provider(settings):
        raise EmbeddingProviderError("provider unavailable")

    monkeypatch.setattr(
        "recall.evaluation.__main__.build_embedding_provider",
        fake_build_provider,
    )
    monkeypatch.setattr(
        "recall.evaluation.__main__.validate_evaluation_state",
        lambda database_path: None,
    )
    monkeypatch.setattr("sys.argv", ["recall.evaluation", str(FIXTURE_PATH)])

    assert evaluation_main() == 1

    captured = capsys.readouterr()
    assert "Evaluation failed: provider unavailable" in captured.out
