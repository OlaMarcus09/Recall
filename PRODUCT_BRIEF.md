# Recall — Product Brief

## One sentence

Recall helps a learner in a learning community recover knowledge that
already exists, without searching through old messages and documents
or asking the same question again.

## The problem

Learning communities already generate valuable knowledge.

Students ask questions. Someone explains a difficult concept. A useful
resource is shared. A lecturer clarifies something. Course PDFs contain
the official material.

But that knowledge is fragmented across documents and conversations.

As time passes, useful answers become difficult to recover. The same
questions are asked again. Students search through old messages or give
up. Group admins and active students become human search engines,
repeatedly finding material and answering questions that have already
been addressed.

The problem is not always that the knowledge does not exist.

Often, the community has forgotten where it is.

## First user

For the first implementation, Recall is built around a student in a
university course community who needs to recover an answer from shared
learning material.

The initial real-world context is a 400-level Business Management study
community at Miva Open University.

This is the first proving ground, not the limit of the product.

## Initial data source

The first version uses real course PDFs supplied to the system.

Later versions may incorporate other community knowledge such as
approved discussion transcripts, notes, slides and other shared
learning material.

## What Recall does

Recall gives a learning community a memory layer over material it
already has.

A learner can ask a natural-language question.

Recall:

1. searches the community's indexed learning material;
2. retrieves the evidence most relevant to the question;
3. answers only from supported evidence;
4. shows the source document and page or location;
5. distinguishes between supported, partially supported and unsupported
   questions;
6. refuses to invent an answer when sufficient evidence is unavailable.

A later vertical slice records questions from learners and groups
semantically similar questions so moderators can see concepts that
people repeatedly struggle with.

## Core product principle

Recall should prefer saying:

"I couldn't find enough information in the material I have."

over producing a plausible but unsupported answer.

Trust is more important than answer coverage.

## The two product moments

### 1. Memory

A learner asks about something buried inside existing course material.

Recall finds it, answers the question and shows where the answer came
from.

### 2. Collective confusion

Different learners ask differently worded questions about the same
underlying concept.

Recall recognizes the pattern and surfaces that concept as an area the
community may be struggling with.

For example:

"17 questions this week were related to break-even analysis."

The first moment proves Recall can remember.

The second proves Recall can help a community understand what it is
forgetting or struggling to learn.

## First vertical slice

The first build proves only the core memory-and-trust loop:

PDF
→ ingestion
→ indexed knowledge
→ learner question
→ retrieval
→ grounded answer
→ citation or explicit refusal

The recurring-confusion capability remains part of the capstone but is
built only after this loop works reliably.

## What Recall is not

Recall is not:

- a general-purpose chatbot;
- an unrestricted AI tutor;
- a replacement for lecturers or learning communities;
- a system that invents course knowledge;
- a content-generation product;
- a WhatsApp scraping tool.

Its job is to recover and organize knowledge that a community already
has.

## Explicitly out of scope for the first vertical slice

- WhatsApp integration
- Telegram integration
- multiple institutions
- multilingual support
- complex authentication or permissions
- gamification
- notifications
- career or opportunity matching
- advanced moderator analytics

These may become later interfaces or capabilities, but they must not
delay proof of the core retrieval loop.

## Expansion path

If the first implementation works, the same underlying system can
support:

- university study groups;
- professional exam-preparation communities;
- bootcamps;
- training cohorts;
- workplace learning communities;
- other groups that repeatedly learn from shared material.

Miva is the first proof, not the ceiling.

## Moonshot demo promise

Within a few minutes, a judge should understand one idea:

Recall remembers what a learning community already knows and refuses
to pretend it knows what the community has never taught it.