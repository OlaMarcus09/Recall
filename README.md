# Recall

Recall is a learning-memory system. The first vertical slice will prove
PDF ingestion, provenance-preserving knowledge representation, semantic
retrieval, grounded answers, visible citations, and explicit refusal
when evidence is insufficient.

This repository currently includes the T01 application foundation and
T02 PDF ingestion with page-level provenance. It does not implement
chunking, embeddings, retrieval, answer generation, citations,
evaluation logic, authentication, or recurring-confusion detection.

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

No secrets are required for T01.
