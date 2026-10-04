from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from recall import db
from recall.retrieval.embeddings import (
    EmbeddedText,
    EmbeddingProvider,
    embed_document,
    source_text_hash,
)
from recall.retrieval.math import cosine_similarity


class IndexingError(RuntimeError):
    """Raised when KnowledgeUnits cannot be represented for retrieval."""


@dataclass(frozen=True)
class IndexingResult:
    material_id: int
    provider: str
    model: str
    vector_dimension: int
    knowledge_unit_count: int


def index_material_embeddings(
    database_path: Path,
    material_id: int,
    provider: EmbeddingProvider,
) -> IndexingResult:
    with db.connect(database_path) as connection:
        units = db.list_knowledge_units_with_materials_from_connection(
            connection,
            material_id,
        )

    if not units:
        raise IndexingError(f"Material {material_id} has no KnowledgeUnits to index.")

    embedded_units: list[tuple[int, str, str]] = []
    vector_dimension: int | None = None

    for unit in units:
        embedded = embed_document(provider, unit["text"])
        validate_provider_identity(embedded, provider)
        validate_embedding_vector(embedded.vector)
        if cosine_similarity(embedded.vector, embedded.vector) is None:
            raise IndexingError("Embedding vector must be finite and non-zero.")
        if vector_dimension is None:
            vector_dimension = embedded.dimension
        elif embedded.dimension != vector_dimension:
            raise IndexingError("Embedding provider returned mixed vector dimensions.")

        embedded_units.append(
            (
                unit["knowledge_unit_id"],
                encode_vector(embedded.vector),
                source_text_hash(unit["text"]),
            )
        )

    if vector_dimension is None:
        raise IndexingError("No embeddings were produced.")

    with db.connect(database_path) as connection:
        with connection:
            db.replace_embeddings_for_material(
                connection,
                material_id=material_id,
                provider=provider.provider,
                model=provider.model,
                vector_dimension=vector_dimension,
                embeddings=embedded_units,
            )

    return IndexingResult(
        material_id=material_id,
        provider=provider.provider,
        model=provider.model,
        vector_dimension=vector_dimension,
        knowledge_unit_count=len(embedded_units),
    )


def validate_provider_identity(
    embedded: EmbeddedText,
    provider: EmbeddingProvider,
) -> None:
    if embedded.provider != provider.provider or embedded.model != provider.model:
        raise IndexingError("Embedding provider returned mismatched identity.")


def validate_embedding_vector(vector: list[float]) -> None:
    if not vector:
        raise IndexingError("Embedding vector cannot be empty.")


def encode_vector(vector: list[float]) -> str:
    return json.dumps(vector, separators=(",", ":"))


def decode_vector(vector_json: str) -> list[float] | None:
    try:
        decoded = json.loads(vector_json)
    except json.JSONDecodeError:
        return None

    if not isinstance(decoded, list):
        return None

    vector: list[float] = []
    for value in decoded:
        if not isinstance(value, int | float):
            return None
        vector.append(float(value))
    return vector
