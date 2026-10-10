# Recall Demo Script

This script is designed for a 3-5 minute capstone presentation. It uses
the committed synthetic Northstar Bakery PDF so judges can reproduce the
demo without private BUA 401 material.

The real product problem came from repeated questions in a Miva study
community. Northstar Bakery is fictional demo data that safely shows the
same product loop: trusted learning memory, grounded citations, refusal
and recurring Community Confusion.

## Setup Before Presenting

Use the README quick demo commands:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
cp .env.example .env

export RECALL_DATABASE_PATH=data/recall-demo.sqlite3
export RECALL_EMBEDDING_PROVIDER=fastembed
export RECALL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
python -m recall.demo prepare demo/recall-demo-material.pdf

python -m uvicorn recall.main:app --reload
```

Open `http://127.0.0.1:8000/`.

The first FastEmbed run may download and cache the local embedding model.
Do that before the live presentation if possible.

For the cleanest Community Confusion count, use a fresh ignored database
path for the live presentation. If you have already used
`data/recall-demo.sqlite3`, choose another local path such as
`data/recall-demo-live.sqlite3` before running `prepare`.

## 1. Real Learner and Problem (about 35 seconds)

Say:

"Recall started from a simple study-community problem. Learners already
have course PDFs and shared explanations, but the knowledge gets buried.
The same questions come back because the community cannot easily recover
what it already knows. Recall is a memory layer for that learning
material."

Show:

- Recall home page.
- Product statement: "Learning that stays with your community."
- Ask Recall form and Community Confusion area.

Expected behavior:

- Page loads.
- No private material or local paths are visible.

Fallback if the app is not running:

"If the live app is unavailable, the README shows the exact local setup
and the committed synthetic PDF used for this demo. The repository also
includes automated tests for the same trust boundaries."

## 2. Existing Learning Material (about 35 seconds)

Say:

"For the public demo, I use a fictional Northstar Bakery handbook. In
the real project, the proving ground was authorized course material from
a Miva study context, but private material is not distributed in the
public repository."

Show:

- README quick demo section or terminal command:
  `python -m recall.demo prepare demo/recall-demo-material.pdf`
- Mention the prepare command performs ingestion, KnowledgeUnit creation
  and embedding indexing.

Expected behavior:

- Prepared demo database has 4 PDF pages, KnowledgeUnits and indexed
  embeddings.

Fallback:

"The command is covered by `tests/test_demo.py`, using fake embeddings so
tests do not require network or private PDFs."

## 3. Grounded Answer and Visible Citation (about 55 seconds)

Ask:

`What are Northstar Bakery's strategic priorities?`

Say:

"Recall does not just answer. It shows the support state and the source
page it used. Page numbers come from the ingested PDF page metadata, not
from generated text."

Show:

- Status: `SUPPORTED` / "Supported by your material".
- Answer content.
- Source card with document title and physical PDF page.
- Evidence excerpt.

Expected behavior:

- A supported answer appears.
- Citation points to the synthetic demo material.
- Physical page 2 should appear for the strategic-priorities question.

Fallback:

"If the live embedding model fails, the expected behavior is still
covered in tests: supported answers require validated citations, and
citations resolve from the database."

## 4. Refusal When Evidence Is Insufficient (about 45 seconds)

Ask:

`Which medication should an employee take for a bacterial infection?`

Say:

"Trust is more important than coverage. If Recall cannot ground an
answer in the indexed material, it refuses instead of using general model
knowledge."

Show:

- Status: `NOT_FOUND` / "Not enough evidence".
- Stable refusal: "I couldn't find enough information in the material I
  have."
- No citations.
- No medical answer.

Expected behavior:

- Recall refuses.
- No source cards are shown.

Fallback:

"This refusal path is tested across the trust layer and web interface.
The product principle is to prefer an honest no-answer over a plausible
unsupported answer."

## 5. Recurring Community Confusion (about 45 seconds)

Ask the pair:

1. `What is Northstar Bakery's mission?`
2. `How would you describe the mission of Northstar Bakery?`

Say:

"Recall also observes anonymous question patterns. Similar questions can
be grouped into Community Confusion so a course community can see what
keeps coming up. It counts repeated questions, not unique people."

Show:

- Community Confusion card.
- Representative question.
- "Asked 2 times" or similar recurrence wording.

Expected behavior:

- If the local FastEmbed similarity crosses the grouping threshold, the
  pair appears as one recurring cluster.
- In a fresh database where the mission question has not already been
  asked earlier in the demo, the recurrence should be "Asked 2 times".
- The UI does not claim "2 students" or expose embeddings.

Fallback:

"If this pair does not group on a specific machine, the grouping policy
is intentionally conservative. The deterministic grouping behavior is
covered by fake-embedding tests, and the README documents the threshold
as an MVP heuristic."

## 6. Codex Development Process and Next Steps (about 45 seconds)

Say:

"Codex helped build Recall as a sequence of small, reviewed checkpoints:
PDF provenance, KnowledgeUnits, retrieval, trust states, citations,
evaluation, Community Confusion, UI and finally reproducible demo
bootstrap. The human decisions were the original problem, the
refusal-first principle, feature scope, visual acceptance and feature
freeze."

Show:

- Git commit history or README "Built with Codex" section.
- Optional: test result summary.

Expected behavior:

- The repository shows incremental commits from T01 through T10.
- Tests pass locally.

Fallback:

"Even without the live demo, the repository contains the full
implementation, the synthetic demo material, and the test suite proving
the core trust boundaries."

## Closing Statement

"Recall is not trying to be a chatbot that knows everything. It is a
memory layer for what a learning community has already provided. It can
answer with sources, refuse when evidence is missing, and surface
questions the community keeps asking. With more time, I would add
production auth, richer answer generation, OCR and calibrated evaluation
on more real material."
