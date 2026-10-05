from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recall import db
from recall.generation.service import SupportStatus
from recall.retrieval.embeddings import EmbeddingProvider, embed_query, source_text_hash
from recall.retrieval.index import decode_vector, encode_vector
from recall.retrieval.math import cosine_similarity


QUESTION_GROUPING_THRESHOLD = 0.82
RECURRING_CONFUSION_MIN_COUNT = 2


class QuestionMemoryError(RuntimeError):
    """Raised when anonymous question memory cannot be recorded safely."""


@dataclass(frozen=True)
class RecordedQuestion:
    question_id: int
    cluster_id: int
    similarity: float | None


@dataclass(frozen=True)
class ConfusionCluster:
    cluster_id: int
    representative_question: str
    recurrence_count: int
    first_seen: str
    last_seen: str


def record_question_memory(
    *,
    database_path: Path,
    question: str,
    support_status: SupportStatus,
    embedding_provider: EmbeddingProvider,
    threshold: float = QUESTION_GROUPING_THRESHOLD,
) -> RecordedQuestion:
    from recall.answering import normalize_question

    normalized_question = normalize_question(question)
    embedded = embed_query(embedding_provider, normalized_question)
    if embedded.provider != embedding_provider.provider or embedded.model != embedding_provider.model:
        raise QuestionMemoryError("Question embedding provider identity mismatch.")
    if not embedded.vector:
        raise QuestionMemoryError("Question embedding vector cannot be empty.")

    db.initialize_database(database_path)
    with db.connect(database_path) as connection:
        with connection:
            cluster_match = find_best_cluster(
                connection,
                provider=embedded.provider,
                model=embedded.model,
                vector=embedded.vector,
                threshold=threshold,
            )
            question_id = insert_learner_question(
                connection,
                normalized_question=normalized_question,
                support_status=support_status,
            )
            insert_question_embedding(
                connection,
                question_id=question_id,
                provider=embedded.provider,
                model=embedded.model,
                vector=embedded.vector,
                source_hash=source_text_hash(normalized_question),
            )

            if cluster_match is None:
                cluster_id = insert_cluster(
                    connection,
                    question_id=question_id,
                    provider=embedded.provider,
                    model=embedded.model,
                    vector=embedded.vector,
                )
                similarity = None
            else:
                cluster_id = cluster_match[0]
                similarity = cluster_match[1]
                assign_question_to_cluster(connection, question_id, cluster_id)
                refresh_cluster_centroid(connection, cluster_id)

            if cluster_match is None:
                assign_question_to_cluster(connection, question_id, cluster_id)

    return RecordedQuestion(
        question_id=question_id,
        cluster_id=cluster_id,
        similarity=similarity,
    )


def find_best_cluster(
    connection,
    *,
    provider: str,
    model: str,
    vector: list[float],
    threshold: float,
) -> tuple[int, float] | None:
    rows = connection.execute(
        """
        SELECT id, centroid_vector_json, vector_dimension
        FROM confusion_clusters
        WHERE provider = ?
          AND model = ?
          AND vector_dimension = ?
        ORDER BY id ASC
        """,
        (provider, model, len(vector)),
    ).fetchall()

    best: tuple[int, float] | None = None
    for row in rows:
        centroid = decode_vector(row["centroid_vector_json"])
        if centroid is None:
            continue
        similarity = cosine_similarity(vector, centroid)
        if similarity is None:
            continue
        if best is None or similarity > best[1] or (
            similarity == best[1] and row["id"] < best[0]
        ):
            best = (row["id"], similarity)

    if best is None or best[1] < threshold:
        return None
    return best


def insert_learner_question(
    connection,
    *,
    normalized_question: str,
    support_status: SupportStatus,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO learner_questions (normalized_text, support_status)
        VALUES (?, ?)
        """,
        (normalized_question, support_status.value),
    )
    return int(cursor.lastrowid)


def insert_question_embedding(
    connection,
    *,
    question_id: int,
    provider: str,
    model: str,
    vector: list[float],
    source_hash: str,
) -> None:
    connection.execute(
        """
        INSERT INTO question_embeddings (
            learner_question_id, provider, model, vector_json,
            vector_dimension, source_text_sha256
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (question_id, provider, model, encode_vector(vector), len(vector), source_hash),
    )


def insert_cluster(
    connection,
    *,
    question_id: int,
    provider: str,
    model: str,
    vector: list[float],
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO confusion_clusters (
            representative_question_id, provider, model,
            vector_dimension, centroid_vector_json
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (question_id, provider, model, len(vector), encode_vector(vector)),
    )
    return int(cursor.lastrowid)


def assign_question_to_cluster(connection, question_id: int, cluster_id: int) -> None:
    connection.execute(
        """
        UPDATE learner_questions
        SET confusion_cluster_id = ?
        WHERE id = ?
        """,
        (cluster_id, question_id),
    )


def refresh_cluster_centroid(connection, cluster_id: int) -> None:
    rows = connection.execute(
        """
        SELECT qe.vector_json
        FROM learner_questions AS lq
        JOIN question_embeddings AS qe
          ON qe.learner_question_id = lq.id
        WHERE lq.confusion_cluster_id = ?
        ORDER BY lq.id ASC
        """,
        (cluster_id,),
    ).fetchall()
    vectors = [decode_vector(row["vector_json"]) for row in rows]
    valid_vectors = [vector for vector in vectors if vector is not None]
    if not valid_vectors:
        raise QuestionMemoryError("Cluster has no valid question embeddings.")
    dimension = len(valid_vectors[0])
    if any(len(vector) != dimension for vector in valid_vectors):
        raise QuestionMemoryError("Cluster question embeddings have mixed dimensions.")

    centroid = [
        sum(vector[index] for vector in valid_vectors) / len(valid_vectors)
        for index in range(dimension)
    ]
    connection.execute(
        """
        UPDATE confusion_clusters
        SET centroid_vector_json = ?
        WHERE id = ?
        """,
        (encode_vector(centroid), cluster_id),
    )


def list_recurring_confusions(
    database_path: Path,
    *,
    min_count: int = RECURRING_CONFUSION_MIN_COUNT,
) -> list[ConfusionCluster]:
    db.initialize_database(database_path)
    with db.connect(database_path) as connection:
        rows = connection.execute(
            """
            SELECT
                cc.id AS cluster_id,
                representative.normalized_text AS representative_question,
                COUNT(lq.id) AS recurrence_count,
                MIN(lq.created_at) AS first_seen,
                MAX(lq.created_at) AS last_seen
            FROM confusion_clusters AS cc
            JOIN learner_questions AS representative
              ON representative.id = cc.representative_question_id
            JOIN learner_questions AS lq
              ON lq.confusion_cluster_id = cc.id
            GROUP BY cc.id, representative.normalized_text
            HAVING COUNT(lq.id) >= ?
            ORDER BY recurrence_count DESC, last_seen DESC, cc.id ASC
            """,
            (min_count,),
        ).fetchall()

    return [
        ConfusionCluster(
            cluster_id=row["cluster_id"],
            representative_question=row["representative_question"],
            recurrence_count=row["recurrence_count"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
        )
        for row in rows
    ]
