from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = 3


def connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: Path) -> None:
    with connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_versions (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO schema_versions (version)
            VALUES (?)
            """,
            (1,),
        )
        applied_version = current_schema_version_from_connection(connection) or 0
        if applied_version < 2:
            apply_schema_v2(connection)
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_versions (version)
                VALUES (?)
                """,
                (2,),
            )
        if applied_version < 3:
            apply_schema_v3(connection)
            connection.execute(
                """
                INSERT OR IGNORE INTO schema_versions (version)
                VALUES (?)
                """,
                (3,),
            )


def current_schema_version_from_connection(
    connection: sqlite3.Connection,
) -> int | None:
    table_exists = connection.execute(
        """
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name = 'schema_versions'
        """
    ).fetchone()
    if table_exists is None:
        return None

    row = connection.execute(
        "SELECT MAX(version) AS version FROM schema_versions"
    ).fetchone()
    return row["version"] if row is not None else None


def apply_schema_v2(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS materials (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            source_type TEXT NOT NULL,
            uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            content_sha256 TEXT NOT NULL UNIQUE,
            page_count INTEGER NOT NULL CHECK (page_count > 0)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS extracted_pages (
            id INTEGER PRIMARY KEY,
            material_id INTEGER NOT NULL,
            page_number INTEGER NOT NULL CHECK (page_number > 0),
            text TEXT NOT NULL,
            FOREIGN KEY (material_id)
                REFERENCES materials (id)
                ON DELETE CASCADE,
            UNIQUE (material_id, page_number)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_extracted_pages_material_order
        ON extracted_pages (material_id, page_number)
        """
    )


def apply_schema_v3(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_extracted_pages_provenance
        ON extracted_pages (id, material_id, page_number)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_units (
            id INTEGER PRIMARY KEY,
            material_id INTEGER NOT NULL,
            extracted_page_id INTEGER NOT NULL,
            page_number INTEGER NOT NULL CHECK (page_number > 0),
            chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
            text TEXT NOT NULL CHECK (length(trim(text)) > 0),
            FOREIGN KEY (material_id)
                REFERENCES materials (id)
                ON DELETE CASCADE,
            FOREIGN KEY (extracted_page_id)
                REFERENCES extracted_pages (id)
                ON DELETE CASCADE,
            FOREIGN KEY (extracted_page_id, material_id, page_number)
                REFERENCES extracted_pages (id, material_id, page_number)
                ON DELETE CASCADE,
            UNIQUE (material_id, extracted_page_id, chunk_index)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_knowledge_units_material_order
        ON knowledge_units (material_id, page_number, chunk_index)
        """
    )


def current_schema_version(database_path: Path) -> int | None:
    with connect(database_path) as connection:
        return current_schema_version_from_connection(connection)


def material_count(database_path: Path) -> int:
    with connect(database_path) as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM materials").fetchone()
        return int(row["count"])


def page_count(database_path: Path) -> int:
    with connect(database_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM extracted_pages"
        ).fetchone()
        return int(row["count"])


def knowledge_unit_count(database_path: Path) -> int:
    with connect(database_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM knowledge_units"
        ).fetchone()
        return int(row["count"])


def get_material_by_hash(
    connection: sqlite3.Connection,
    content_sha256: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT id, title, original_filename, source_type, uploaded_at,
               content_sha256, page_count
        FROM materials
        WHERE content_sha256 = ?
        """,
        (content_sha256,),
    ).fetchone()


def insert_material(
    connection: sqlite3.Connection,
    *,
    title: str,
    original_filename: str,
    source_type: str,
    content_sha256: str,
    page_count_value: int,
) -> sqlite3.Row:
    cursor = connection.execute(
        """
        INSERT INTO materials (
            title, original_filename, source_type, content_sha256, page_count
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (title, original_filename, source_type, content_sha256, page_count_value),
    )
    material_id = cursor.lastrowid
    row = connection.execute(
        """
        SELECT id, title, original_filename, source_type, uploaded_at,
               content_sha256, page_count
        FROM materials
        WHERE id = ?
        """,
        (material_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError("Inserted material could not be loaded.")
    return row


def insert_extracted_pages(
    connection: sqlite3.Connection,
    *,
    material_id: int,
    pages: Iterable[tuple[int, str]],
) -> None:
    connection.executemany(
        """
        INSERT INTO extracted_pages (material_id, page_number, text)
        VALUES (?, ?, ?)
        """,
        [(material_id, page_number, text) for page_number, text in pages],
    )


def list_material_pages(
    database_path: Path,
    material_id: int,
) -> list[sqlite3.Row]:
    with connect(database_path) as connection:
        return list(
            connection.execute(
                """
                SELECT id, material_id, page_number, text
                FROM extracted_pages
                WHERE material_id = ?
                ORDER BY page_number ASC
                """,
                (material_id,),
            ).fetchall()
        )


def material_exists(connection: sqlite3.Connection, material_id: int) -> bool:
    row = connection.execute(
        "SELECT 1 FROM materials WHERE id = ?",
        (material_id,),
    ).fetchone()
    return row is not None


def list_material_pages_from_connection(
    connection: sqlite3.Connection,
    material_id: int,
) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            """
            SELECT id, material_id, page_number, text
            FROM extracted_pages
            WHERE material_id = ?
            ORDER BY page_number ASC
            """,
            (material_id,),
        ).fetchall()
    )


def replace_knowledge_units_for_material(
    connection: sqlite3.Connection,
    *,
    material_id: int,
    units: Iterable[tuple[int, int, int, str]],
) -> None:
    connection.execute(
        "DELETE FROM knowledge_units WHERE material_id = ?",
        (material_id,),
    )
    connection.executemany(
        """
        INSERT INTO knowledge_units (
            material_id, extracted_page_id, page_number, chunk_index, text
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (material_id, extracted_page_id, page_number, chunk_index, text)
            for extracted_page_id, page_number, chunk_index, text in units
        ],
    )


def list_knowledge_units_for_material(
    database_path: Path,
    material_id: int,
) -> list[sqlite3.Row]:
    with connect(database_path) as connection:
        return list_knowledge_units_for_material_from_connection(
            connection,
            material_id,
        )


def list_knowledge_units_for_material_from_connection(
    connection: sqlite3.Connection,
    material_id: int,
) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            """
            SELECT id, material_id, extracted_page_id, page_number,
                   chunk_index, text
            FROM knowledge_units
            WHERE material_id = ?
            ORDER BY page_number ASC, chunk_index ASC
            """,
            (material_id,),
        ).fetchall()
    )


def list_materials(database_path: Path) -> list[sqlite3.Row]:
    with connect(database_path) as connection:
        return list(
            connection.execute(
                """
                SELECT id, title, original_filename, source_type, uploaded_at,
                       content_sha256, page_count
                FROM materials
                ORDER BY uploaded_at ASC, id ASC
                """
            ).fetchall()
        )
