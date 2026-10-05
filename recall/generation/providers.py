from __future__ import annotations

import re
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
    """A deterministic fallback that mechanically composes supplied evidence."""

    provider = "extractive"
    model = "bounded-evidence-excerpts"
    max_evidence_items = 3
    max_answer_chars = 1100

    def generate(
        self,
        *,
        question: str,
        evidence: list[EvidenceItem],
    ) -> GeneratedAnswer:
        if not evidence:
            raise GenerationProviderError("No evidence was supplied.")

        excerpts: list[tuple[int, str]] = []
        seen_fingerprints: set[str] = set()
        for item in evidence:
            normalized = normalize_whitespace(item.text)
            if not normalized:
                continue
            fingerprint = evidence_fingerprint(normalized)
            if fingerprint and fingerprint in seen_fingerprints:
                continue

            label = f"Evidence {len(excerpts) + 1}: "
            remaining = remaining_excerpt_budget(
                excerpts=excerpts,
                next_label=label,
                max_answer_chars=self.max_answer_chars,
            )
            if remaining <= 0:
                break

            excerpt = bounded_excerpt(normalized, remaining)
            if not excerpt:
                break
            excerpts.append((item.knowledge_unit_id, excerpt))
            if fingerprint:
                seen_fingerprints.add(fingerprint)
            if len(excerpts) >= self.max_evidence_items:
                break

        if not excerpts:
            raise GenerationProviderError("No usable evidence was supplied.")

        cited_ids = [knowledge_unit_id for knowledge_unit_id, _ in excerpts]
        snippet = " ".join(
            f"Evidence {index}: {excerpt}"
            for index, (_, excerpt) in enumerate(excerpts, start=1)
        )
        return GeneratedAnswer(
            answer_text=f"The available material says: {snippet}",
            cited_knowledge_unit_ids=cited_ids,
        )


def remaining_excerpt_budget(
    *,
    excerpts: list[tuple[int, str]],
    next_label: str,
    max_answer_chars: int,
) -> int:
    used = sum(
        len(f"Evidence {index}: {excerpt}")
        for index, (_, excerpt) in enumerate(excerpts, start=1)
    )
    if excerpts:
        used += len(" ") * len(excerpts)
    return max_answer_chars - used - len(next_label)


def bounded_excerpt(text: str, max_chars: int) -> str:
    normalized = normalize_whitespace(text)
    if len(normalized) <= max_chars:
        return normalized

    cutoff = normalized.rfind(". ", 0, max_chars)
    if cutoff >= max_chars // 2:
        return normalized[: cutoff + 1].rstrip()
    return normalized[: max_chars - 3].rstrip() + "..."


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def evidence_fingerprint(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(words[:48])
