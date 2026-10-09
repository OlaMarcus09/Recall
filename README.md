# Recall

Recall is a learning-memory system. The first vertical slice will prove
PDF ingestion, provenance-preserving knowledge representation, semantic
retrieval, grounded answers, visible citations, and explicit refusal
when evidence is insufficient.

This repository currently includes the T01 application foundation, T02
PDF ingestion with page-level provenance, and T03 page-scoped
KnowledgeUnit chunking, and T04 semantic retrieval through persisted
embeddings. It also includes the T05A provider-independent grounded
answer and trust layer, the minimal learner-facing UI, a trust
evaluation harness, and anonymous recurring-confusion grouping. It does
not include a real LLM provider, authentication, user profiles, or a
polished analytics dashboard.

## Requirements

- Python 3.11 or 3.12

FastEmbed currently depends on ONNXRuntime wheels that may not resolve
reliably on Python 3.13 in this local development environment. Use
Python 3.11 or 3.12 for the local embedding provider.

## Setup

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
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

## Quick Demo From a Fresh Clone

Recall includes a small synthetic PDF so reviewers can prepare a working
local demo without private course material. The document is fictional
and exists only to demonstrate ingestion, citations, refusal and
recurring question grouping.

From a fresh clone:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env
```

Prepare a demo database from the included synthetic PDF:

```bash
export RECALL_DATABASE_PATH=data/recall-demo.sqlite3
export RECALL_EMBEDDING_PROVIDER=fastembed
export RECALL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
python -m recall.demo prepare demo/recall-demo-material.pdf
```

The first FastEmbed run may download and cache
`BAAI/bge-small-en-v1.5`, so initial preparation can take longer and may
require internet access. After the model is cached, Recall uses the
local provider without an OpenAI API key.

Start the web app:

```bash
python -m uvicorn recall.main:app --reload
```

Then open `http://127.0.0.1:8000/` and try:

- Supported: `What is Northstar Bakery's mission?`
- Supported: `What are Northstar Bakery's strategic priorities?`
- Unsupported: `Which medication should an employee take for a bacterial infection?`
- Confusion pair:
  - `What is Northstar Bakery's mission?`
  - `How would you describe the mission of Northstar Bakery?`

The supported questions should return grounded answers with source
citations and physical PDF pages. The unsupported question should return
Recall's refusal instead of a fabricated answer. Similar repeated
questions can appear in Community Confusion as anonymous recurrence,
reported as "Asked N times" rather than unique learner counts.

To prepare your own authorized text-extractable PDF, use the same
command shape:

```bash
python -m recall.demo prepare /path/to/authorized-course-material.pdf
```

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
The default local provider/model configuration is FastEmbed
`BAAI/bge-small-en-v1.5`, which does not require an OpenAI API key.
OpenAI `text-embedding-3-small` remains available as an optional
provider when a valid `OPENAI_API_KEY` is configured. Tests use offline
deterministic fakes rather than live providers.

Indexing a material embeds all current KnowledgeUnits before replacing
that material's stored embeddings in one transaction. Retrieval computes
the question embedding, filters to a compatible provider/model/vector
dimension, and ranks valid rows with cosine similarity. Result provenance
is joined from persisted KnowledgeUnit, ExtractedPage and Material data.

FastEmbed may download/cache the model on first use. After the model is
cached locally, Recall can index and retrieve with the FastEmbed provider
without an API key or OpenAI request. Do not commit model cache files.

## Grounded Answers

T05A turns ranked retrieval evidence into a service-level response with:

- `SUPPORTED`
- `PARTIAL`
- `NOT_FOUND`

The generation provider is deliberately narrow: it receives only the
learner question and selected KnowledgeUnit text, then returns answer
text plus KnowledgeUnit IDs it claims to cite. Recall treats provider
output as untrusted. Citation IDs are valid only when they were supplied
as evidence and still resolve through the database to authoritative
Material, page and chunk provenance.

Citation validation proves grounding, not semantic completeness. Recall
keeps retrieval selection, evidence sufficiency, and citation grounding
as separate decisions so unused retrieval candidates do not automatically
make an answer `PARTIAL`.

The current implementation is fully offline-testable with deterministic
fake providers and includes a simple extractive provider for development
demonstrations. It does not install Ollama, call OpenAI, download a local
LLM, or persist generated answer records. When the learner web/API flow
uses question memory, Recall stores normalized question text, the actual
support status returned by Recall, question embedding metadata and a
confusion-cluster assignment for recurring-question grouping. It does
not store generated answers, citation excerpts or user identity fields
for that memory layer; question text itself may still contain personal
information and should be treated as sensitive local data.

## Learner UI

The root page provides the minimal learner question flow. It runs the
same retrieval and grounded-answer service used by tests, then renders:

- the answer or stable `NOT_FOUND` refusal;
- support status;
- citations with database-derived document title, physical PDF page and
  chunk index;
- a short bounded excerpt from each cited KnowledgeUnit.

The web layer does not implement retrieval, generation, confidence logic
or provenance resolution. It calls the application service and renders the
returned citation contract. Citation page numbers are physical 1-based PDF
pages persisted during ingestion, not generated or inferred labels.

## Evaluation Harness

T07 adds a developer-facing trust evaluation harness:

```bash
python -m recall.evaluation path/to/fixture.json
```

The fixture contains stable case IDs, learner questions and expected
support states. Optional expected concepts and reference notes are
metadata for human review only; they are not passed into retrieval,
generation or prompts.

The harness reports machine-checkable trust metrics such as status
matches, false supported answers, false refusals, citation presence,
NOT_FOUND citation absence, provenance validity and page-number
availability. Semantic answer quality remains a human-review field
(`CORRECT`, `INCORRECT`, or `UNREVIEWED`); Recall does not use keyword
matching or an LLM judge to pretend semantic grading is automatic.

Use synthetic fixtures in the repository. Real course PDFs, extracted
text, evaluation databases, embeddings and private benchmark reports
must remain local and ignored.

## Collective Confusion

T08 adds anonymous question memory. After Recall answers a validated
learner question, it can store the normalized question text, the actual
support status returned by Recall, a separate question embedding, and a
confusion-cluster assignment. This memory is observational: it does not
change retrieval, generation, support states or citations.

Question embeddings are stored separately from course KnowledgeUnit
embeddings. They include provider, model, vector dimension and source
hash metadata so incompatible question representations are not compared.

Grouping uses a deterministic MVP heuristic: compare the new question
embedding with existing cluster centroids for the same provider/model
and dimension, then join the closest cluster only when cosine similarity
is at least `0.82`. This threshold is intentionally conservative and
should be calibrated with more real evaluation data. The first question
in a cluster is the representative question; it is a label for human
inspection, not an LLM-generated concept name.

Recurring confusions are clusters with at least two persisted questions.
Recurrence counts are derived from question membership, so repeated
legitimate submissions count as community signal. The JSON endpoint
`GET /api/confusions` returns cluster summaries only; it does not expose
embeddings or learner identity.

Question text may contain personal information typed by a learner, so it
should be treated as sensitive local data. Recall does not store user
accounts, names, emails, phone numbers, IP addresses, device IDs or
per-learner profiles for this slice.

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
- `RECALL_EMBEDDING_PROVIDER`: embedding provider name, default `fastembed`
- `RECALL_EMBEDDING_MODEL`: embedding model, default `BAAI/bge-small-en-v1.5`
- `RECALL_FASTEMBED_CACHE_DIR`: optional FastEmbed model cache path
- `OPENAI_API_KEY`: required only for the live OpenAI embedding provider

Do not commit real API keys or private course material.
