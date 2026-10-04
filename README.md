# Recall

Recall is a learning-memory system. The first vertical slice will prove
PDF ingestion, provenance-preserving knowledge representation, semantic
retrieval, grounded answers, visible citations, and explicit refusal
when evidence is insufficient.

This repository currently includes the T01 application foundation, T02
PDF ingestion with page-level provenance, and T03 page-scoped
KnowledgeUnit chunking, and T04 semantic retrieval through persisted
embeddings. It does not implement answer generation, citations,
evaluation scoring, authentication, or recurring-confusion detection.

## Requirements

- Python 3.11+

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the application and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Create local configuration if needed:

```bash
cp .env.example .env
```

The default configuration stores local runtime data under `data/` and
uploads under `uploads/`. These paths are ignored by Git and must not
contain committed private course material.

## PDF Ingestion

T02 supports authorized, text-extractable PDFs through PyMuPDF. Ingestion
creates one `Material` record and one `extracted_pages` record per source
page. Product-facing page numbers are 1-based and come from the PDF page
order, not from document text.

Duplicate ingestion is detected by SHA-256 of the PDF bytes. Reingesting
identical bytes returns the existing `Material` instead of creating a
duplicate, even if the filename changes.

Scanned/image-only PDFs, malformed PDFs, encrypted PDFs, empty files, and
non-PDF inputs are rejected. Failed ingestion is atomic and should not
leave partial material or page records behind.

## Knowledge Units

T03 converts each extracted page into deterministic `KnowledgeUnit`
records. Chunking is page-scoped: a chunk never combines text from
multiple PDF pages. Product-facing page numbers remain 1-based and are
copied from `extracted_pages`, not derived from text.

The chunker normalizes whitespace by collapsing runs of whitespace to a
single space, then uses paragraph and sentence boundaries where possible.
Chunks target 700 characters with a 900-character hard maximum. Running
generation again for the same material replaces that material's existing
KnowledgeUnits instead of duplicating them.

## Semantic Retrieval

T04 stores embeddings as derived data linked to `KnowledgeUnit` records.
The default provider/model configuration is OpenAI
`text-embedding-3-small`, but tests use an offline deterministic fake
provider.

Indexing a material embeds all current KnowledgeUnits before replacing
that material's stored embeddings in one transaction. Retrieval computes
the question embedding, filters to a compatible provider/model/vector
dimension, and ranks valid rows with cosine similarity. Result provenance
is joined from persisted KnowledgeUnit, ExtractedPage and Material data.

## Run

```bash
python -m uvicorn recall.main:app --reload
```

Then open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`

## Test

```bash
python -m pytest
```

## Configuration

Recall reads configuration from environment variables:

- `RECALL_ENV`: runtime environment label, default `development`
- `RECALL_DATABASE_PATH`: SQLite database path, default `data/recall.sqlite3`
- `RECALL_UPLOAD_DIR`: future upload storage path, default `uploads`
- `RECALL_EMBEDDING_PROVIDER`: embedding provider name, default `openai`
- `RECALL_EMBEDDING_MODEL`: embedding model, default `text-embedding-3-small`
- `OPENAI_API_KEY`: required only for the live OpenAI embedding provider

Do not commit real API keys or private course material.
