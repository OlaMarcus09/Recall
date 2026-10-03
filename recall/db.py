from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_VERSION = 1


def connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
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
            (SCHEMA_VERSION,),
        )


def current_schema_version(database_path: Path) -> int | None:
    with connect(database_path) as connection:
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
