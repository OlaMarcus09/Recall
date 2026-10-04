from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EmbeddedText:
    vector: list[float]
    provider: str
    model: str

    @property
    def dimension(self) -> int:
        return len(self.vector)


class EmbeddingProvider(Protocol):
    provider: str
    model: str

    def embed(self, text: str) -> EmbeddedText:
        """Return an embedding for text without exposing provider response objects."""


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider cannot produce a trustworthy vector."""


class OpenAIEmbeddingProvider:
    provider = "openai"

    def __init__(self, *, api_key: str | None, model: str) -> None:
        if not api_key:
            raise EmbeddingProviderError("OPENAI_API_KEY is required.")
        self.model = model
        self._api_key = api_key

    def embed(self, text: str) -> EmbeddedText:
        from openai import OpenAI, OpenAIError

        client = OpenAI(api_key=self._api_key)
        try:
            response = client.embeddings.create(
                model=self.model,
                input=text,
            )
        except OpenAIError:
            raise EmbeddingProviderError(
                "OpenAI embedding request failed."
            ) from None
        vector = [float(value) for value in response.data[0].embedding]
        return EmbeddedText(vector=vector, provider=self.provider, model=self.model)


def source_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
