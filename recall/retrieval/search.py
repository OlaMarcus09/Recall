from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recall import db
from recall.retrieval.embeddings import EmbeddingProvider, source_text_hash
from recall.retrieval.index import decode_vector
from recall.retrieval.math import cosine_similarity


MAX_TOP_K = 20


class RetrievalError(ValueError):
    """Raised when retrieval cannot produce trustworthy ranked evidence."""


@dataclass(frozen=True)
class RetrievalResult:
    rank: int
    knowledge_unit_id: int
    similarity: float
    text: str
    material_id: int
    material_title: str
    original_filename: str
    extracted_page_id: int
    page_number: int
    chunk_index: int


def retrieve_relevant_knowledge_units(
    database_path: Path,
    *,
    question: str,
    provider: EmbeddingProvider,
    top_k: int = 5,
    material_id: int | None = None,
) -> list[RetrievalResult]:
    query = question.strip()
    if not query:
        raise RetrievalError("Question text cannot be empty.")
    validate_top_k(top_k)

    query_embedding = provider.embed(query)
    if query_embedding.provider != provider.provider or query_embedding.model != provider.model:
        raise RetrievalError("Query embedding provider identity mismatch.")

    with db.connect(database_path) as connection:
        if material_id is not None:
            ensure_complete_material_index(
                connection,
                material_id=material_id,
                provider=provider.provider,
                model=provider.model,
                vector_dimension=query_embedding.dimension,
            )
        rows = db.list_embedding_rows_for_search(
            connection,
            provider=provider.provider,
            model=provider.model,
            vector_dimension=query_embedding.dimension,
            material_id=material_id,
        )

    scored: list[tuple[float, object]] = []
    for row in rows:
        if row["source_text_sha256"] != source_text_hash(row["text"]):
            continue
        vector = decode_vector(row["vector_json"])
        if vector is None:
            continue
        similarity = cosine_similarity(query_embedding.vector, vector)
        if similarity is None:
            continue
        scored.append((similarity, row))

    if material_id is not None and not scored:
        raise RetrievalError("No valid embeddings are available for retrieval.")

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1]["material_id"],
            item[1]["page_number"],
            item[1]["chunk_index"],
            item[1]["knowledge_unit_id"],
        )
    )

    return [
        RetrievalResult(
            rank=rank,
            knowledge_unit_id=row["knowledge_unit_id"],
            similarity=similarity,
            text=row["text"],
            material_id=row["material_id"],
            material_title=row["material_title"],
            original_filename=row["original_filename"],
            extracted_page_id=row["extracted_page_id"],
            page_number=row["page_number"],
            chunk_index=row["chunk_index"],
        )
        for rank, (similarity, row) in enumerate(scored[:top_k], start=1)
    ]


def ensure_complete_material_index(
    connection,
    *,
    material_id: int,
    provider: str,
    model: str,
    vector_dimension: int,
) -> None:
    index = db.get_material_embedding_index(
        connection,
        material_id=material_id,
        provider=provider,
        model=model,
    )
    if index is None:
        raise RetrievalError("Material has not been indexed for this provider/model.")
    if index["vector_dimension"] != vector_dimension:
        raise RetrievalError("Indexed vector dimension does not match the query.")

    rows = db.list_embedding_rows_for_search(
        connection,
        provider=provider,
        model=model,
        vector_dimension=vector_dimension,
        material_id=material_id,
    )
    valid_count = sum(
        1
        for row in rows
        if row["source_text_sha256"] == source_text_hash(row["text"])
    )
    if valid_count != index["knowledge_unit_count"]:
        raise RetrievalError("Material embedding index is stale or incomplete.")


def validate_top_k(top_k: int) -> None:
    if top_k < 1:
        raise RetrievalError("top_k must be at least 1.")
    if top_k > MAX_TOP_K:
        raise RetrievalError(f"top_k cannot exceed {MAX_TOP_K}.")
