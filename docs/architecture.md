# Recall Architecture Notes

## T01 Scope

T01 establishes the smallest runnable foundation for Recall:

- FastAPI application shell
- Jinja2-rendered home page
- environment-based configuration
- SQLite schema lifecycle bootstrap
- health endpoint
- pytest smoke tests

The implementation intentionally does not include PDF extraction,
chunking, embeddings, semantic retrieval, answer generation, confidence
states, citations, evaluation logic, authentication, messaging
integrations, or recurring-confusion detection.

## Current Storage

SQLite is initialized locally using `RECALL_DATABASE_PATH`. T01 creates
only a `schema_versions` table so later tasks have a clear migration
boundary without pretending that domain persistence exists yet.

Local database files, uploads, private PDFs, course material, and local
evaluation artifacts are ignored by Git.
