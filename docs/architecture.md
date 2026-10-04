# Recall Architecture Notes

## T01 Scope

T01 establishes the smallest runnable foundation for Recall:

- FastAPI application shell
- Jinja2-rendered home page
- environment-based configuration
- SQLite schema lifecycle bootstrap
- health endpoint
- pytest smoke tests

The T01 foundation did not include answer generation, confidence states,
citations, evaluation scoring, authentication, messaging integrations,
or recurring-confusion detection. Later slices add retrieval and the
provider-independent grounded-answer trust layer while still avoiding
final learner UI and recurring-confusion behavior.

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

## T04 Semantic Retrieval

T04 stores embeddings separately from source/provenance data:

- `knowledge_unit_embeddings`: derived vectors for KnowledgeUnits
- `material_embedding_indexes`: completed material/provider/model index
  metadata

Embedding records include provider, model, vector dimension and a hash
of the KnowledgeUnit text they were derived from. Retrieval only uses
embeddings matching the configured provider/model and query dimension.
If KnowledgeUnit text changes without re-indexing, the stale embedding
is excluded and material-scoped retrieval treats the index as incomplete.

Cosine similarity is computed in Python over SQLite-loaded vectors. The
ranking layer returns evidence only: score, KnowledgeUnit text, Material
metadata, extracted page id, page number and chunk index. It does not
generate answers or classify confidence states.

## T04.1 Local Embeddings

Recall supports two embedding providers behind the same narrow
`EmbeddingProvider` boundary:

- `fastembed` with `BAAI/bge-small-en-v1.5` for local no-key retrieval
- `openai` with `text-embedding-3-small` when a valid API key is present

FastEmbed is the default development provider. It may download/cache the
model on first use, then can run locally without an OpenAI request. Model
caches are local artifacts and must remain outside Git.

The local FastEmbed path should run on Python 3.11 or 3.12. In the
current development environment, FastEmbed's ONNXRuntime dependency did
not resolve reliably under Python 3.13.

For BGE/FastEmbed, KnowledgeUnit text is embedded as passages and learner
questions are embedded as queries. This keeps BGE's query/document
retrieval convention inside the provider layer and leaves persisted
KnowledgeUnit source text unchanged.

OpenAI and FastEmbed embeddings coexist by provider, model and vector
dimension. Selecting a different provider/model requires indexing under
that identity; retrieval never compares vectors from incompatible
providers or models.

## T05A Grounded Answer Trust Layer

T05A adds a generation boundary without adding a real LLM runtime. The
`GenerationProvider` contract accepts a learner question and selected
evidence items containing only KnowledgeUnit IDs and text. It returns
answer text and cited KnowledgeUnit IDs. It does not own support states,
page numbers, titles, filenames, chunk indexes or citation display
metadata.

Recall keeps three separate concepts apart:

- Retrieval selection chooses candidate evidence for generation.
- Evidence sufficiency/coverage represents whether Recall has a known
  reason to believe the available evidence only partially supports the
  requested answer.
- Citation grounding validates that the generated answer cites only
  supplied evidence and that citations resolve to database provenance.

Recall owns the trust policy:

- If retrieved evidence is absent or too weak for a conservative first
  pass, the service returns `NOT_FOUND` without calling generation.
- If generation fails, is malformed, returns no citation, or cites a
  KnowledgeUnit that was not supplied as evidence, the service returns
  `NOT_FOUND`.
- If generation succeeds, at least one citation validates against the
  database, and no known partial-support signal exists, the response is
  `SUPPORTED`.
- If generation succeeds with valid citations but Recall has an explicit
  known partial-support assessment, the response is `PARTIAL` and the
  answer is prefixed to state that the material only partially supports
  the response.

This policy is deterministic and intentionally conservative. It uses a
small low-evidence gate and a relative top-score band to select evidence,
but it does not treat any single cosine score as proof that an answer is
supported. It also does not infer answer completeness from the number or
percentage of selected retrieval chunks cited. Calibration and richer
semantic coverage checks belong to later evaluation work.

Citation validation is database-derived. A citation is valid only when
the cited KnowledgeUnit ID was supplied to the provider and still resolves
through persisted `knowledge_units` and `materials` rows. Any page,
document or chunk metadata attempted by a provider is ignored. Citation
validity proves grounding, not semantic completeness.

Prompt construction treats retrieved text as untrusted data. Evidence is
placed in explicit KnowledgeUnit blocks and the instructions tell the
provider to answer only from supplied evidence, cite only supplied IDs,
avoid outside knowledge, and ignore instructions embedded inside course
material.

T05A includes deterministic fake-provider testing and a simple extractive
development provider. It does not persist questions/answers, install
Ollama, call OpenAI generation APIs, add a local LLM, tune retrieval, or
introduce T06 recurring-confusion behavior.
