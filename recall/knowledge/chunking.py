from __future__ import annotations

import re


TARGET_CHARS = 700
MAX_CHARS = 900


def chunk_text(
    text: str,
    *,
    target_chars: int = TARGET_CHARS,
    max_chars: int = MAX_CHARS,
) -> list[str]:
    if target_chars <= 0:
        raise ValueError("target_chars must be positive.")
    if max_chars < target_chars:
        raise ValueError("max_chars must be greater than or equal to target_chars.")

    segments = split_source_text(text, max_chars=max_chars)
    chunks: list[str] = []
    current = ""

    for segment in segments:
        if not current:
            current = segment
            continue

        candidate = f"{current} {segment}"
        if len(candidate) <= target_chars:
            current = candidate
            continue

        chunks.append(current)
        current = segment

    if current:
        chunks.append(current)

    return chunks


def split_source_text(text: str, *, max_chars: int) -> list[str]:
    paragraphs = [
        normalize_chunk_text(paragraph)
        for paragraph in re.split(r"\n\s*\n+", text)
    ]
    segments: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if len(paragraph) <= max_chars:
            segments.append(paragraph)
            continue

        for sentence in split_sentences(paragraph):
            if len(sentence) <= max_chars:
                segments.append(sentence)
            else:
                segments.extend(split_long_segment(sentence, max_chars=max_chars))

    return segments


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def split_long_segment(text: str, *, max_chars: int) -> list[str]:
    words = text.split(" ")
    chunks: list[str] = []
    current = ""

    for word in words:
        if not word:
            continue
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_word(word, max_chars=max_chars))
            continue

        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        chunks.append(current)
        current = word

    if current:
        chunks.append(current)

    return chunks


def split_long_word(word: str, *, max_chars: int) -> list[str]:
    return [
        word[index : index + max_chars]
        for index in range(0, len(word), max_chars)
    ]


def normalize_chunk_text(text: str) -> str:
    return " ".join(text.split())
