# Recall — Capstone Specification

## Status

Build-week specification.

This document defines what Recall must prove before additional
interfaces or features are added.

---

## Goal

Prove that a learner can ask a natural-language question and receive a
grounded, cited answer from learning material already provided to
Recall — and that Recall refuses to invent an answer when sufficient
evidence is unavailable.

---

## Primary user

A student participating in a learning community that already has shared
course material but has difficulty recovering useful information from
it later.

Initial proving context:

400-level Business Management study material from Miva Open University.

---

## First vertical slice

The complete path is:

1. An admin uploads a text-extractable PDF.
2. Recall extracts text while preserving document and page provenance.
3. The extracted material is divided into retrievable units.
4. Those units are indexed for semantic retrieval.
5. A learner enters a natural-language question through a minimal web
   interface.
6. Recall retrieves the most relevant evidence.
7. Recall determines whether the available evidence is sufficient.
8. If sufficient, Recall generates an answer grounded only in that
   evidence.
9. The response displays the source document and page/location.
10. If evidence is insufficient, Recall explicitly says that it could
    not find enough information.

No WhatsApp integration is required for this slice.

---

## Grounding rule

Retrieved material is evidence, not merely additional context.

The answer generator must not introduce factual claims that are not
supported by the retrieved evidence.

Every substantive answer must retain provenance back to the source
material.

When evidence is insufficient, Recall must refuse rather than fill the
gap using general model knowledge.

---

## Confidence states

Every response belongs to one of three states.

### SUPPORTED

The retrieved material provides sufficient evidence to answer the
question.

Recall answers and cites the supporting source.

### PARTIAL

Relevant material exists, but it does not completely support the
requested answer.

Recall may summarize what the material does support, but must explicitly
state the limitation.

### NOT_FOUND

The indexed material does not provide sufficient evidence.

Recall does not answer from model knowledge.

It returns a clear message such as:

"I couldn't find enough information in the material I have."

Thresholds or classification rules used to produce these states must be
documented and testable rather than silently chosen and forgotten.

---

## Citation requirements

Every SUPPORTED or PARTIAL answer must include provenance.

At minimum:

- source document;
- page number or source location;
- supporting retrieved chunk/reference.

Page numbers must originate from ingestion metadata.

The model must never generate or infer a page number itself.

NOT_FOUND responses do not require a citation.

---

## Initial data source

### Supported

Text-extractable PDF course material.

### Assumptions

- PDFs contain extractable text.
- English is sufficient for build week.
- Uploaded documents are authorized for use in the demo.
- The first slice can operate with a small number of documents.
- Large-scale concurrency is unnecessary for the demonstration.

### Known limitation

Scanned/image-only PDFs requiring OCR are outside the first slice.

---

## Minimum domain model

### Material

- id
- title
- original_filename
- source_type
- uploaded_at

### KnowledgeUnit

- id
- material_id
- page_or_location
- text
- retrieval representation
- ordering/provenance metadata

### Question

- id
- text
- created_at
- retrieved_knowledge_unit_ids
- confidence_state

### Answer

- id
- question_id
- response_text
- confidence_state
- cited_knowledge_unit_ids

### ConceptCluster — Slice 2

- id
- label
- related_question_ids
- question_count
- related_material_ids
- created_at / updated_at

The exact database/storage implementation is intentionally not mandated
by this specification. The implementation plan must justify that
choice.

---

## Evaluation set

Before judging retrieval quality, create a fixed evaluation set from
real demo material.

The set should contain at least:

- 5 direct-answer questions;
- 5 paraphrased questions;
- 3 questions requiring retrieval from less obvious sections;
- 5 deliberately unsupported questions.

The expected source page/location for supported questions must be
recorded before evaluation.

Do not rewrite the evaluation questions after seeing Recall's results
simply to improve the score.

---

## Acceptance criteria

### AC1 — Ingestion provenance

Given a supported PDF, Recall extracts usable text while preserving the
correct source document and page/location metadata.

Verification:
Manually inspect samples from the beginning, middle and end of an
ingested document.

### AC2 — Grounded retrieval

For the fixed supported-question evaluation set, Recall retrieves
evidence from the expected source page/location for at least 80% of
questions.

Retrieval quality and answer quality should be measured separately.

### AC3 — Citation integrity

Every SUPPORTED and PARTIAL response contains a citation whose document
and page/location exist in the ingested data.

A fabricated page/location is a failure.

### AC4 — Refusal behaviour

For deliberately unsupported questions, Recall must not present an
unsupported answer as if it came from the indexed material.

The expected outcome is NOT_FOUND or an explicitly limited PARTIAL
response where appropriate.

### AC5 — New-material portability

A second supported PDF can be ingested and queried without changing
application code.

### AC6 — Regression protection

Automated tests cover the critical trust boundaries, including:

- provenance preservation;
- citation presence;
- invalid citation prevention;
- unsupported-question refusal;
- confidence-state response shape.

---

## Build tasks

### T01 — Repository and application foundation

Establish the smallest working application structure, development
commands, test setup and configuration boundaries.

No retrieval implementation yet.

### T02 — PDF ingestion with provenance

Accept a supported PDF and extract text while preserving document and
page metadata.

Prove provenance before continuing.

### T03 — Knowledge indexing

Split extracted material into retrieval units and create a persistent
searchable representation.

The implementation plan must justify chunking strategy and storage
choice.

### T04 — Retrieval

Given a question, return ranked relevant knowledge units with their
provenance.

Test retrieval independently from answer generation.

### T05 — Grounded answer and confidence states

Generate responses using retrieved evidence only.

Implement SUPPORTED, PARTIAL and NOT_FOUND behaviour.

### T06 — Citation contract and minimal learner UI

Expose the complete learner path:

question
→ response
→ confidence state
→ source citation

Keep the interface intentionally small.

### T07 — Evaluation and trust tests

Create the fixed evaluation dataset and automated regression tests.

Document failures honestly rather than adjusting tests to match current
behaviour.

### T08 — Collective confusion

After T01–T07 are working:

- persist learner questions;
- compare semantic similarity between questions;
- group related questions;
- count recurrence;
- surface recurring concepts in a minimal moderator view.

This task is part of the capstone roadmap, not an optional product idea,
but it must not block completion of the first vertical slice.

---

## Slice 2 — Collective confusion

The goal of Slice 2 is to answer a different question:

"What is this community repeatedly struggling with?"

Example:

Three learners ask:

- "What's the difference between fixed and variable cost?"
- "I don't understand which costs change with production."
- "Why does variable cost increase when output increases?"

Recall should be capable of associating these with the same underlying
concept rather than treating them as unrelated strings.

The first implementation does not need sophisticated analytics.

A simple demonstrable output is enough:

"Variable vs fixed costs — 7 related questions."

---

## Demo path

The demo must be reproducible.

### Beat 1 — Existing knowledge

Show a real course PDF already ingested into Recall.

### Beat 2 — Ask

Ask a real question whose answer exists in the material.

### Beat 3 — Remember

Recall returns a concise answer.

### Beat 4 — Prove

Open or show the cited document/page so the evidence can be verified.

### Beat 5 — Trust

Ask a deliberately unsupported question.

Recall responds that it cannot find sufficient evidence rather than
inventing an answer.

### Beat 6 — Community memory

If Slice 2 is complete, show several differently worded questions being
recognized as one recurring area of confusion.

---

## Explicit non-goals for Slice 1

Do not build:

- WhatsApp integration;
- Telegram integration;
- advanced authentication;
- institution management;
- multiple roles unless technically required;
- OCR;
- multilingual retrieval;
- notifications;
- gamification;
- elaborate dashboards;
- unrelated AI tutoring functionality.

---

## Definition of first-slice complete

The first vertical slice is complete when:

- a real PDF can be ingested;
- provenance survives ingestion and retrieval;
- a real learner question retrieves useful evidence;
- the generated answer is grounded in that evidence;
- the learner can see where the answer came from;
- unsupported questions are handled safely;
- the critical trust behaviours have tests;
- the complete path can be demonstrated reliably.