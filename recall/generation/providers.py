from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EvidenceItem:
    knowledge_unit_id: int
    text: str


@dataclass(frozen=True)
class GeneratedAnswer:
    answer_text: str
    cited_knowledge_unit_ids: list[int]


class GenerationProvider(Protocol):
    provider: str
    model: str

    def generate(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
    ) -> GeneratedAnswer:
        """Generate answer text with citations to supplied KnowledgeUnit IDs only."""


class GenerationProviderError(RuntimeError):
    """Raised when a generation provider cannot produce a safe response."""


class ExtractiveGenerationProvider:
    """A deterministic fallback that mechanically quotes supplied evidence."""

    provider = "extractive"
    model = "first-evidence-snippet"

    def generate(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
    ) -> GeneratedAnswer:
        if not evidence:
            raise GenerationProviderError("No evidence was supplied.")
        first = evidence[0]
        snippet = first.text.strip()
        if len(snippet) > 280:
            snippet = snippet[:277].rstrip() + "..."
        return GeneratedAnswer(
            answer_text=f"The available material says: {snippet}",
            cited_knowledge_unit_ids=[first.knowledge_unit_id],
        )
