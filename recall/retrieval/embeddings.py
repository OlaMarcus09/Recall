from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


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


def embed_document(provider: EmbeddingProvider, text: str) -> EmbeddedText:
    method = getattr(provider, "embed_document", None)
    if method is not None:
        return method(text)
    return provider.embed(text)


def embed_query(provider: EmbeddingProvider, text: str) -> EmbeddedText:
    method = getattr(provider, "embed_query", None)
    if method is not None:
        return method(text)
    return provider.embed(text)


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


class FastEmbedEmbeddingProvider:
    provider = "fastembed"

    def __init__(
        self,
        *,
        model: str,
        cache_dir: Path | None = None,
        embedding_class: type[Any] | None = None,
    ) -> None:
        self.model = model
        try:
            if embedding_class is None:
                from fastembed import TextEmbedding

                embedding_class = TextEmbedding
            self._embedding_model = embedding_class(
                model_name=model,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
        except Exception:
            raise EmbeddingProviderError(
                "FastEmbed model could not be loaded. Ensure the model is available "
                "locally or allow the initial FastEmbed download."
            ) from None

    def embed(self, text: str) -> EmbeddedText:
        return self.embed_document(text)

    def embed_document(self, text: str) -> EmbeddedText:
        try:
            vector = next(iter(self._embedding_model.passage_embed([text])))
        except Exception:
            raise EmbeddingProviderError("FastEmbed document embedding failed.") from None
        return self._embedded_text_from_vector(vector)

    def embed_query(self, text: str) -> EmbeddedText:
        try:
            vector = next(iter(self._embedding_model.query_embed(text)))
        except Exception:
            raise EmbeddingProviderError("FastEmbed query embedding failed.") from None
        return self._embedded_text_from_vector(vector)

    def _embedded_text_from_vector(self, vector: Any) -> EmbeddedText:
        values = vector.tolist() if hasattr(vector, "tolist") else vector
        if not isinstance(values, list) or not values:
            raise EmbeddingProviderError("FastEmbed returned an invalid vector.")

        converted: list[float] = []
        for value in values:
            if not isinstance(value, int | float):
                raise EmbeddingProviderError("FastEmbed returned an invalid vector.")
            float_value = float(value)
            if not math.isfinite(float_value):
                raise EmbeddingProviderError("FastEmbed returned an invalid vector.")
            converted.append(float_value)

        return EmbeddedText(vector=converted, provider=self.provider, model=self.model)


def build_embedding_provider(settings) -> EmbeddingProvider:
    provider_name = settings.embedding_provider.strip().lower()
    if provider_name == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )
    if provider_name == "fastembed":
        return FastEmbedEmbeddingProvider(
            model=settings.embedding_model,
            cache_dir=settings.fastembed_cache_dir,
        )
    raise EmbeddingProviderError(
        f"Unsupported embedding provider: {settings.embedding_provider}"
    )


def source_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
