from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recall import db
from recall.ingestion.service import IngestionResult, ingest_pdf
from recall.knowledge.service import KnowledgeUnit, generate_knowledge_units_for_material
from recall.retrieval.embeddings import EmbeddingProvider
from recall.retrieval.index import IndexingResult, index_material_embeddings


class DemoPreparationError(RuntimeError):
    """Raised when demo preparation cannot complete safely."""


@dataclass(frozen=True)
class DemoPreparationResult:
    ingestion: IngestionResult
    knowledge_units: list[KnowledgeUnit]
    index: IndexingResult
    database_path: Path


def prepare_demo_material(
    *,
    database_path: Path,
    pdf_path: Path,
    embedding_provider: EmbeddingProvider,
    title: str | None = None,
) -> DemoPreparationResult:
    if not pdf_path.is_file():
        raise DemoPreparationError(f"PDF file does not exist: {pdf_path}")

    db.initialize_database(database_path)
    try:
        pdf_bytes = pdf_path.read_bytes()
    except OSError as exc:
        raise DemoPreparationError(f"Could not read PDF file: {pdf_path}") from exc

    ingestion = ingest_pdf(
        database_path=database_path,
        pdf_bytes=pdf_bytes,
        original_filename=pdf_path.name,
        title=title,
    )
    knowledge_units = generate_knowledge_units_for_material(
        database_path,
        ingestion.material.id,
    )
    index = index_material_embeddings(
        database_path,
        ingestion.material.id,
        embedding_provider,
    )

    if index.knowledge_unit_count != len(knowledge_units):
        raise DemoPreparationError(
            "Indexed KnowledgeUnit count does not match generated KnowledgeUnits."
        )

    return DemoPreparationResult(
        ingestion=ingestion,
        knowledge_units=knowledge_units,
        index=index,
        database_path=database_path,
    )
