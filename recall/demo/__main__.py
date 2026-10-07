from __future__ import annotations

import argparse
import sqlite3
from dataclasses import replace
from pathlib import Path

from recall.config import Settings
from recall.demo.service import DemoPreparationError, prepare_demo_material
from recall.retrieval.embeddings import EmbeddingProviderError, build_embedding_provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m recall.demo",
        description="Prepare a Recall demo database from an authorized PDF.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Ingest, chunk and index a text-extractable PDF.",
    )
    prepare_parser.add_argument("pdf", type=Path, help="Path to authorized PDF material.")
    prepare_parser.add_argument(
        "--title",
        help="Optional display title for the material.",
    )
    prepare_parser.add_argument(
        "--database",
        type=Path,
        help="Override RECALL_DATABASE_PATH for this preparation run.",
    )

    args = parser.parse_args(argv)
    if args.command != "prepare":
        parser.error("Unsupported command.")

    settings = Settings.from_env()
    if args.database is not None:
        settings = replace(settings, database_path=args.database)

    try:
        provider = build_embedding_provider(settings)
        result = prepare_demo_material(
            database_path=settings.database_path,
            pdf_path=args.pdf,
            embedding_provider=provider,
            title=args.title,
        )
    except (
        DemoPreparationError,
        EmbeddingProviderError,
        OSError,
        RuntimeError,
        sqlite3.Error,
        ValueError,
    ) as exc:
        print(f"Recall demo preparation failed: {exc}")
        return 1

    print_success(result)
    return 0


def print_success(result) -> None:
    material = result.ingestion.material
    index = result.index
    created_label = "created" if result.ingestion.created else "already existed"
    print("Recall demo prepared.")
    print(f"Material: {material.title} ({created_label})")
    print(f"Pages: {material.page_count}")
    print(f"Knowledge units: {len(result.knowledge_units)}")
    print(f"Embedding provider: {index.provider}")
    print(f"Model: {index.model}")
    print(f"Indexed units: {index.knowledge_unit_count}")
    print(f"Database: {result.database_path}")
    print("")
    print("Next:")
    print("python -m uvicorn recall.main:app --reload")


if __name__ == "__main__":
    raise SystemExit(main())
