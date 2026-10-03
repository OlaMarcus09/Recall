# Recall Architecture Notes

## T01 Scope

T01 establishes the smallest runnable foundation for Recall:

- FastAPI application shell
- Jinja2-rendered home page
- environment-based configuration
- SQLite schema lifecycle bootstrap
- health endpoint
- pytest smoke tests

The implementation intentionally does not include embeddings, semantic
retrieval, answer generation, confidence states, citations, evaluation
logic, authentication, messaging integrations, or recurring-confusion
detection.

## Current Storage

SQLite is initialized locally using `RECALL_DATABASE_PATH`. The
`schema_versions` table keeps the build-week schema lifecycle explicit
without introducing a separate migration framework.

Local database files, uploads, private PDFs, course material, and local
evaluation artifacts are ignored by Git.

## T02 PDF Ingestion

T02 adds document-level and page-level persistence only:

- `materials`: source identity and document metadata
- `extracted_pages`: exact page text with 1-based source page numbers

T02 does not create retrieval chunks or embedding-ready knowledge units.
Chunking belongs to T03.

Duplicate ingestion is prevented by a SHA-256 hash of the PDF bytes.
Filenames are stored only as sanitized display metadata and are never
used as durable source identity or filesystem paths.

## T03 Knowledge Units

T03 adds page-scoped `knowledge_units` records. Each unit stores:

- `material_id`
- `extracted_page_id`
- 1-based `page_number`
- per-page `chunk_index`
- normalized source text

The database enforces that a KnowledgeUnit's material, extracted page,
and page number match the same `extracted_pages` row. The chunking
service processes one extracted page at a time, so chunks cannot cross
PDF page boundaries.

Chunking is deterministic and inspectable: whitespace is collapsed,
paragraph/sentence boundaries are preferred, chunks target 700
characters, and chunks have a 900-character hard maximum. Re-running
generation for a material replaces that material's existing
KnowledgeUnits inside one transaction.
