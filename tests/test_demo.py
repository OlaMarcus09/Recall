from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz
import pytest

from recall import db
from recall.demo.__main__ import main
from recall.demo.service import DemoPreparationError, prepare_demo_material
from recall.retrieval.embeddings import EmbeddedText, EmbeddingProviderError


@dataclass
class FakeDemoEmbeddingProvider:
    provider: str = "fake-demo"
    model: str = "fake-demo-model"
    fail: bool = False

    def embed(self, text: str) -> EmbeddedText:
        if self.fail:
            raise RuntimeError("simulated embedding failure")
        lowered = text.lower()
        vector = [
            float(lowered.count("northstar") + 1),
            float(lowered.count("mission") + 1),
            float(len(lowered) % 19 + 1),
        ]
        return EmbeddedText(vector=vector, provider=self.provider, model=self.model)


def make_pdf(page_texts: list[str]) -> bytes:
    document = fitz.open()
    try:
        for page_text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), page_text)
        return document.tobytes()
    finally:
        document.close()


def write_pdf(path: Path, page_texts: list[str]) -> Path:
    path.write_bytes(make_pdf(page_texts))
    return path


def test_prepare_valid_pdf_ingests_chunks_and_indexes(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    pdf_path = write_pdf(
        tmp_path / "demo.pdf",
        [
            "Northstar mission material for a synthetic demo.",
            "Northstar priorities remain page-scoped for citations.",
        ],
    )

    result = prepare_demo_material(
        database_path=database_path,
        pdf_path=pdf_path,
        embedding_provider=FakeDemoEmbeddingProvider(),
        title="Synthetic Demo",
    )

    assert result.ingestion.created is True
    assert result.ingestion.material.title == "Synthetic Demo"
    assert result.ingestion.material.page_count == 2
    assert len(result.knowledge_units) >= 2
    assert result.index.provider == "fake-demo"
    assert result.index.model == "fake-demo-model"
    assert result.index.knowledge_unit_count == len(result.knowledge_units)
    assert db.material_count(database_path) == 1
    assert db.knowledge_unit_count(database_path) == len(result.knowledge_units)
    assert db.embedding_count(database_path) == len(result.knowledge_units)


def test_prepare_rerun_reuses_existing_material_without_destructive_delete(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    pdf_path = write_pdf(
        tmp_path / "demo.pdf",
        ["Northstar duplicate ingestion should stay idempotent."],
    )
    provider = FakeDemoEmbeddingProvider()

    first = prepare_demo_material(
        database_path=database_path,
        pdf_path=pdf_path,
        embedding_provider=provider,
        title="First Title",
    )
    second = prepare_demo_material(
        database_path=database_path,
        pdf_path=pdf_path,
        embedding_provider=provider,
        title="Second Title",
    )

    assert first.ingestion.created is True
    assert second.ingestion.created is False
    assert second.ingestion.material.id == first.ingestion.material.id
    assert second.ingestion.material.title == "First Title"
    assert db.material_count(database_path) == 1
    assert db.embedding_count(database_path) == len(second.knowledge_units)


def test_prepare_missing_file_fails_clearly(tmp_path):
    with pytest.raises(DemoPreparationError, match="PDF file does not exist"):
        prepare_demo_material(
            database_path=tmp_path / "recall.sqlite3",
            pdf_path=tmp_path / "missing.pdf",
            embedding_provider=FakeDemoEmbeddingProvider(),
        )


def test_prepare_non_pdf_fails_without_partial_records(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    input_path = tmp_path / "not-a-pdf.txt"
    input_path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Only PDF files"):
        prepare_demo_material(
            database_path=database_path,
            pdf_path=input_path,
            embedding_provider=FakeDemoEmbeddingProvider(),
        )

    assert db.material_count(database_path) == 0


def test_prepare_blank_pdf_fails_without_partial_records(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    document = fitz.open()
    try:
        document.new_page()
        pdf_path = tmp_path / "blank.pdf"
        pdf_path.write_bytes(document.tobytes())
    finally:
        document.close()

    with pytest.raises(ValueError, match="no meaningful extractable text"):
        prepare_demo_material(
            database_path=database_path,
            pdf_path=pdf_path,
            embedding_provider=FakeDemoEmbeddingProvider(),
        )

    assert db.material_count(database_path) == 0


def test_prepare_encrypted_pdf_fails_without_partial_records(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    document = fitz.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), "Encrypted synthetic text.")
        pdf_path = tmp_path / "encrypted.pdf"
        pdf_path.write_bytes(
            document.tobytes(
                encryption=fitz.PDF_ENCRYPT_AES_256,
                owner_pw="owner-password",
                user_pw="user-password",
            )
        )
    finally:
        document.close()

    with pytest.raises(ValueError, match="Encrypted PDFs are not supported"):
        prepare_demo_material(
            database_path=database_path,
            pdf_path=pdf_path,
            embedding_provider=FakeDemoEmbeddingProvider(),
        )

    assert db.material_count(database_path) == 0


def test_prepare_provider_failure_does_not_pretend_index_succeeded(tmp_path):
    database_path = tmp_path / "recall.sqlite3"
    pdf_path = write_pdf(tmp_path / "demo.pdf", ["Northstar mission text."])

    with pytest.raises(RuntimeError, match="simulated embedding failure"):
        prepare_demo_material(
            database_path=database_path,
            pdf_path=pdf_path,
            embedding_provider=FakeDemoEmbeddingProvider(fail=True),
        )

    assert db.embedding_count(database_path) == 0


def test_cli_success_summary_uses_configured_provider_without_openai(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        "recall.demo.__main__.build_embedding_provider",
        lambda settings: FakeDemoEmbeddingProvider(),
    )
    database_path = tmp_path / "recall.sqlite3"
    pdf_path = write_pdf(
        tmp_path / "demo.pdf",
        ["Northstar mission wording should not be dumped into output."],
    )

    exit_code = main(
        [
            "prepare",
            str(pdf_path),
            "--database",
            str(database_path),
            "--title",
            "CLI Demo",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Recall demo prepared." in output
    assert "Material: CLI Demo (created)" in output
    assert "Embedding provider: fake-demo" in output
    assert "Model: fake-demo-model" in output
    assert "python -m uvicorn recall.main:app --reload" in output
    assert "mission wording" not in output
    assert "[" not in output


def test_cli_provider_initialization_failure_is_actionable(
    tmp_path,
    monkeypatch,
    capsys,
):
    def fail_provider(settings):
        raise EmbeddingProviderError("FastEmbed model could not be loaded.")

    monkeypatch.setattr("recall.demo.__main__.build_embedding_provider", fail_provider)
    pdf_path = write_pdf(tmp_path / "demo.pdf", ["Northstar text."])

    exit_code = main(
        [
            "prepare",
            str(pdf_path),
            "--database",
            str(tmp_path / "recall.sqlite3"),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Recall demo preparation failed" in output
    assert "FastEmbed model could not be loaded" in output


def test_cli_database_failure_is_reported_without_traceback(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        "recall.demo.__main__.build_embedding_provider",
        lambda settings: FakeDemoEmbeddingProvider(),
    )
    pdf_path = write_pdf(tmp_path / "demo.pdf", ["Northstar text."])

    exit_code = main(
        [
            "prepare",
            str(pdf_path),
            "--database",
            str(tmp_path),
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "Recall demo preparation failed" in output
    assert "Traceback" not in output


def test_included_synthetic_demo_pdf_prepares_with_fake_provider(tmp_path):
    pdf_path = Path("demo/recall-demo-material.pdf")
    assert pdf_path.is_file()

    result = prepare_demo_material(
        database_path=tmp_path / "recall.sqlite3",
        pdf_path=pdf_path,
        embedding_provider=FakeDemoEmbeddingProvider(),
    )

    assert result.ingestion.material.title == "recall-demo-material"
    assert result.ingestion.material.page_count == 4
    assert len(result.knowledge_units) >= 4
    assert result.index.knowledge_unit_count == len(result.knowledge_units)
