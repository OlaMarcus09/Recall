from __future__ import annotations

from dataclasses import dataclass

import fitz
import pytest
from fastapi.testclient import TestClient

from recall import db
from recall.answering import LearnerAnswer, LearnerCitation
from recall.confusion.service import ConfusionCluster, record_question_memory
from recall.config import Settings
from recall.generation.service import REFUSAL_MESSAGE, SupportStatus
from recall.main import create_app
from recall.retrieval.embeddings import EmbeddedText
from recall.web.routes import MAX_UPLOAD_BYTES


@dataclass
class RouteFakeEmbeddingProvider:
    provider: str = "fake"
    model: str = "fake-model"

    def embed(self, text: str) -> EmbeddedText:
        return EmbeddedText(vector=[1.0], provider=self.provider, model=self.model)


@pytest.fixture(autouse=True)
def fake_embedding_provider(monkeypatch):
    monkeypatch.setattr(
        "recall.web.routes.build_embedding_provider",
        lambda settings: RouteFakeEmbeddingProvider(),
    )


def make_pdf(page_texts: list[str]) -> bytes:
    document = fitz.open()
    try:
        for page_text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), page_text)
        return document.tobytes()
    finally:
        document.close()


def make_blank_pdf() -> bytes:
    document = fitz.open()
    try:
        document.new_page()
        return document.tobytes()
    finally:
        document.close()


def make_client(tmp_path, client_address=("testclient", 50000)):
    settings = Settings(
        environment="test",
        database_path=tmp_path / "recall.sqlite3",
        upload_dir=tmp_path / "uploads",
    )
    return TestClient(create_app(settings), client=client_address)


def make_local_client(tmp_path):
    return make_client(tmp_path, client_address=("127.0.0.1", 50000))


def citation(**overrides):
    values = {
        "knowledge_unit_id": 101,
        "material_id": 7,
        "document_title": "Synthetic Policy Notes",
        "original_filename": "synthetic-policy.pdf",
        "page_number": 12,
        "chunk_index": 2,
        "evidence_excerpt": "Strategy evidence from the persisted KnowledgeUnit.",
    }
    values.update(overrides)
    return LearnerCitation(**values)


def test_get_learner_page_renders_product_frame_and_form(tmp_path):
    with make_client(tmp_path) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Learning that stays with your community." in response.text
    assert "Learning Memory" in response.text
    assert "Ask Recall" in response.text
    assert "What would you like to understand?" in response.text
    assert 'maxlength="500"' in response.text
    assert "Ask a question about your course material..." in response.text
    assert "Add material" in response.text
    assert 'name="material_pdf"' in response.text
    assert "PDF only, up to 10 MB" in response.text
    assert "Community Confusion" in response.text
    assert "Environment:" not in response.text


def test_local_pdf_upload_ingests_chunks_and_indexes(monkeypatch, tmp_path):
    pdf_bytes = make_pdf(
        [
            "Upload mission evidence for a searchable local material.",
            "Upload strategy evidence remains page scoped.",
        ]
    )

    with make_local_client(tmp_path) as client:
        response = client.post(
            "/materials/upload",
            files={"material_pdf": ("upload-demo.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    assert "Material indexed and ready to search." in response.text
    assert "upload-demo" in response.text
    assert "2 pages" in response.text
    assert db.material_count(tmp_path / "recall.sqlite3") == 1
    assert db.knowledge_unit_count(tmp_path / "recall.sqlite3") >= 2
    assert db.embedding_count(tmp_path / "recall.sqlite3") >= 2
    assert not list((tmp_path / "uploads").glob("recall-upload-*"))


def test_upload_rejects_invalid_file_content(tmp_path):
    with make_local_client(tmp_path) as client:
        response = client.post(
            "/materials/upload",
            files={"material_pdf": ("not-a-pdf.pdf", b"not a pdf", "application/pdf")},
        )

    assert response.status_code == 400
    assert "Upload a valid text-extractable PDF file." in response.text
    assert db.material_count(tmp_path / "recall.sqlite3") == 0


def test_upload_rejects_oversized_pdf(tmp_path):
    oversized = b"%PDF-1.7\n" + (b"x" * (MAX_UPLOAD_BYTES + 1))

    with make_local_client(tmp_path) as client:
        response = client.post(
            "/materials/upload",
            files={"material_pdf": ("large.pdf", oversized, "application/pdf")},
        )

    assert response.status_code == 413
    assert "PDF uploads are limited to 10 MB." in response.text
    assert not list((tmp_path / "uploads").glob("recall-upload-*"))


def test_upload_rejects_scanned_or_image_only_pdf(tmp_path):
    with make_local_client(tmp_path) as client:
        response = client.post(
            "/materials/upload",
            files={
                "material_pdf": ("blank.pdf", make_blank_pdf(), "application/pdf"),
            },
        )

    assert response.status_code == 400
    assert "The uploaded PDF has no extractable text." in response.text
    assert db.material_count(tmp_path / "recall.sqlite3") == 0


def test_duplicate_upload_refreshes_index_without_duplicate_material(tmp_path):
    pdf_bytes = make_pdf(["Duplicate upload should reuse the same material."])

    with make_local_client(tmp_path) as client:
        first = client.post(
            "/materials/upload",
            files={"material_pdf": ("first.pdf", pdf_bytes, "application/pdf")},
        )
        second = client.post(
            "/materials/upload",
            files={"material_pdf": ("renamed.pdf", pdf_bytes, "application/pdf")},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert "Material already existed; search index refreshed." in second.text
    assert db.material_count(tmp_path / "recall.sqlite3") == 1


def test_upload_rejects_non_loopback_client(tmp_path):
    pdf_bytes = make_pdf(["Remote clients should not be allowed to upload."])

    with make_client(tmp_path, client_address=("203.0.113.10", 50000)) as client:
        response = client.post(
            "/materials/upload",
            files={"material_pdf": ("remote.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 403
    assert "available only from this local machine" in response.text
    assert db.material_count(tmp_path / "recall.sqlite3") == 0


def test_upload_indexing_failure_is_safe_and_cleans_temp(monkeypatch, tmp_path):
    def fail_index(*args, **kwargs):
        raise RuntimeError("simulated index failure with /private/path")

    monkeypatch.setattr("recall.web.routes.index_material_embeddings", fail_index)
    pdf_bytes = make_pdf(["Indexing failure should not leave temporary uploads."])

    with make_local_client(tmp_path) as client:
        response = client.post(
            "/materials/upload",
            files={"material_pdf": ("failure.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 500
    assert "Recall could not prepare that PDF for search." in response.text
    assert "/private/path" not in response.text
    assert not list((tmp_path / "uploads").glob("recall-upload-*"))


def test_successful_upload_is_searchable_immediately(tmp_path):
    pdf_bytes = make_pdf(
        [
            "Northstar local upload mission: make course knowledge searchable.",
            "A separate page describes implementation details.",
        ]
    )

    with make_local_client(tmp_path) as client:
        upload = client.post(
            "/materials/upload",
            files={
                "material_pdf": (
                    "answer-ready.pdf",
                    pdf_bytes,
                    "application/pdf",
                )
            },
        )
        answer = client.post(
            "/ask",
            data={"question": "What is the Northstar local upload mission?"},
        )

    assert upload.status_code == 200
    assert answer.status_code == 200
    assert "Supported by your material" in answer.text
    assert "Northstar local upload mission" in answer.text
    assert "answer-ready" in answer.text


def test_supported_response_renders_answer_and_db_citation(monkeypatch, tmp_path):
    def fake_answer_question(**kwargs):
        return LearnerAnswer(
            question=kwargs["question"].strip(),
            status=SupportStatus.SUPPORTED,
            answer="Strategy creates sustainable advantage.",
            citations=[citation()],
        )

    monkeypatch.setattr("recall.web.routes.answer_question", fake_answer_question)

    with make_client(tmp_path) as client:
        response = client.post(
            "/ask",
            data={"question": "  What does strategy create?  "},
        )

    assert response.status_code == 200
    assert "Supported by your material" in response.text
    assert "Strategy creates sustainable advantage." in response.text
    assert "Synthetic Policy Notes" in response.text
    assert "Page 12" in response.text
    assert "Chunk 2" not in response.text
    assert "Strategy evidence from the persisted KnowledgeUnit." in response.text
    assert ">What does strategy create?<" in response.text


def test_multiple_citations_render(monkeypatch, tmp_path):
    def fake_answer_question(**kwargs):
        return LearnerAnswer(
            question=kwargs["question"],
            status=SupportStatus.SUPPORTED,
            answer="Mission and vision are both described.",
            citations=[
                citation(
                    knowledge_unit_id=1,
                    page_number=3,
                    evidence_excerpt="Mission evidence.",
                ),
                citation(
                    knowledge_unit_id=2,
                    page_number=4,
                    evidence_excerpt="Vision evidence.",
                ),
            ],
        )

    monkeypatch.setattr("recall.web.routes.answer_question", fake_answer_question)

    with make_client(tmp_path) as client:
        response = client.post("/ask", data={"question": "Mission vs vision?"})

    assert response.status_code == 200
    assert "Page 3" in response.text
    assert "Page 4" in response.text
    assert "Mission evidence." in response.text
    assert "Vision evidence." in response.text


def test_partial_response_is_visibly_distinguished(monkeypatch, tmp_path):
    def fake_answer_question(**kwargs):
        return LearnerAnswer(
            question=kwargs["question"],
            status=SupportStatus.PARTIAL,
            answer="The available material only partially supports an answer.",
            citations=[citation()],
        )

    monkeypatch.setattr("recall.web.routes.answer_question", fake_answer_question)

    with make_client(tmp_path) as client:
        response = client.post("/ask", data={"question": "Multi-part question?"})

    assert response.status_code == 200
    assert "Partial support" in response.text
    assert "only partially supports" in response.text


def test_not_found_renders_refusal_without_fake_citations(monkeypatch, tmp_path):
    def fake_answer_question(**kwargs):
        return LearnerAnswer(
            question=kwargs["question"],
            status=SupportStatus.NOT_FOUND,
            answer=REFUSAL_MESSAGE,
            citations=[],
        )

    monkeypatch.setattr("recall.web.routes.answer_question", fake_answer_question)

    with make_client(tmp_path) as client:
        response = client.post("/ask", data={"question": "Unsupported?"})

    assert response.status_code == 200
    assert "Not enough evidence" in response.text
    assert "I couldn&#39;t find enough information in the material I have." in response.text
    assert "Synthetic Policy Notes" not in response.text
    assert "Page 12" not in response.text


def test_blank_and_overlong_questions_are_rejected(tmp_path):
    with make_client(tmp_path) as client:
        blank_response = client.post("/ask", data={"question": "   "})
        overlong_response = client.post("/ask", data={"question": "x" * 501})

    assert blank_response.status_code == 400
    assert "Enter a question" in blank_response.text
    assert overlong_response.status_code == 400
    assert "500 characters" in overlong_response.text


def test_html_escaping_protects_question_answer_and_evidence(monkeypatch, tmp_path):
    def fake_answer_question(**kwargs):
        return LearnerAnswer(
            question=kwargs["question"].strip(),
            status=SupportStatus.SUPPORTED,
            answer="<script>alert('answer')</script>",
            citations=[
                citation(
                    document_title="<b>Unsafe Title</b>",
                    evidence_excerpt="<img src=x onerror=alert(1)>",
                )
            ],
        )

    monkeypatch.setattr("recall.web.routes.answer_question", fake_answer_question)

    with make_client(tmp_path) as client:
        response = client.post(
            "/ask",
            data={"question": "<script>alert('question')</script>"},
        )

    assert response.status_code == 200
    assert "<script>alert('answer')</script>" not in response.text
    assert "<script>alert('question')</script>" not in response.text
    assert "<b>Unsafe Title</b>" not in response.text
    assert "<img src=x onerror=alert(1)>" not in response.text
    assert "&lt;script&gt;alert(&#39;answer&#39;)&lt;/script&gt;" in response.text
    assert "&lt;b&gt;Unsafe Title&lt;/b&gt;" in response.text


def test_provider_construction_failure_returns_safe_refusal(monkeypatch, tmp_path):
    def fail_build_provider(settings):
        raise RuntimeError("local model path /secret/internal failed")

    monkeypatch.setattr("recall.web.routes.build_embedding_provider", fail_build_provider)

    with make_client(tmp_path) as client:
        response = client.post("/ask", data={"question": "What is strategy?"})

    assert response.status_code == 200
    assert "Not enough evidence" in response.text
    assert "I couldn&#39;t find enough information in the material I have." in response.text
    assert "/secret/internal" not in response.text


def test_empty_confusion_state_renders(tmp_path):
    with make_client(tmp_path) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "No recurring questions yet." in response.text


def test_recurring_confusion_section_renders_service_result(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "recall.web.routes.list_recurring_confusions",
        lambda database_path: [
            ConfusionCluster(
                cluster_id=1,
                representative_question="How does photosynthesis work?",
                recurrence_count=3,
                first_seen="2026-01-01 00:00:00",
                last_seen="2026-01-01 00:10:00",
            )
        ],
    )

    with make_client(tmp_path) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "How does photosynthesis work?" in response.text
    assert "Asked 3 times" in response.text
    assert "Similar questions are grouped" in response.text
    assert "anonymously" in response.text
    assert "3 students" not in response.text
    assert "3 learners" not in response.text
    assert "embedding" not in response.text.lower()
    assert "vector" not in response.text.lower()


def test_recurring_confusion_question_is_escaped(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "recall.web.routes.list_recurring_confusions",
        lambda database_path: [
            ConfusionCluster(
                cluster_id=1,
                representative_question="<script>alert('cluster')</script>",
                recurrence_count=2,
                first_seen="2026-01-01 00:00:00",
                last_seen="2026-01-01 00:10:00",
            )
        ],
    )

    with make_client(tmp_path) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "<script>alert('cluster')</script>" not in response.text
    assert "&lt;script&gt;alert(&#39;cluster&#39;)&lt;/script&gt;" in response.text


def test_newly_recurring_cluster_appears_after_form_submission(monkeypatch, tmp_path):
    def fake_answer_question(**kwargs):
        result = LearnerAnswer(
            question=kwargs["question"].strip(),
            status=SupportStatus.NOT_FOUND,
            answer=REFUSAL_MESSAGE,
            citations=[],
        )
        if kwargs.get("remember_question"):
            record_question_memory(
                database_path=kwargs["database_path"],
                question=result.question,
                support_status=result.status,
                embedding_provider=kwargs["embedding_provider"],
            )
        return result

    monkeypatch.setattr("recall.web.routes.answer_question", fake_answer_question)

    with make_client(tmp_path) as client:
        first = client.post("/ask", data={"question": "What is photosynthesis?"})
        second = client.post(
            "/ask",
            data={"question": "Can you explain how photosynthesis works?"},
        )

    assert first.status_code == 200
    assert "No recurring questions yet." in first.text
    assert second.status_code == 200
    assert "What is photosynthesis?" in second.text
    assert "Asked 2 times" in second.text


def test_invalid_question_is_not_persisted_or_grouped(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/ask", data={"question": "   "})
        confusions = client.get("/api/confusions")

    assert response.status_code == 400
    assert confusions.json() == {"clusters": []}


def test_json_endpoint_uses_same_response_contract(monkeypatch, tmp_path):
    def fake_answer_question(**kwargs):
        return LearnerAnswer(
            question=kwargs["question"].strip(),
            status=SupportStatus.SUPPORTED,
            answer="Strategy creates advantage.",
            citations=[citation()],
        )

    monkeypatch.setattr("recall.web.routes.answer_question", fake_answer_question)

    with make_client(tmp_path) as client:
        response = client.post("/api/ask", json={"question": "  Strategy?  "})

    assert response.status_code == 200
    assert response.json() == {
        "question": "Strategy?",
        "status": "SUPPORTED",
        "answer": "Strategy creates advantage.",
        "citations": [
            {
                "knowledge_unit_id": 101,
                "material_id": 7,
                "document_title": "Synthetic Policy Notes",
                "original_filename": "synthetic-policy.pdf",
                "page_number": 12,
                "chunk_index": 2,
                "evidence_excerpt": "Strategy evidence from the persisted KnowledgeUnit.",
            }
        ],
    }


def test_json_endpoint_rejects_blank_question(tmp_path):
    with make_client(tmp_path) as client:
        response = client.post("/api/ask", json={"question": " "})

    assert response.status_code == 400
    assert response.json() == {"error": "Enter a question before asking Recall."}


def test_api_confusions_contract_still_omits_embeddings(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "recall.web.routes.list_recurring_confusions",
        lambda database_path: [
            ConfusionCluster(
                cluster_id=5,
                representative_question="Why do plants need light?",
                recurrence_count=2,
                first_seen="2026-01-01 00:00:00",
                last_seen="2026-01-01 00:10:00",
            )
        ],
    )

    with make_client(tmp_path) as client:
        response = client.get("/api/confusions")

    assert response.status_code == 200
    assert response.json() == {
        "clusters": [
            {
                "cluster_id": 5,
                "representative_question": "Why do plants need light?",
                "recurrence_count": 2,
                "first_seen": "2026-01-01 00:00:00",
                "last_seen": "2026-01-01 00:10:00",
            }
        ]
    }
    assert "embedding" not in response.text
    assert "vector" not in response.text
