from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        # Split on sentence-ending patterns, keeping the delimiter with the sentence
        sentences = re.split(r'(?<=\.)\s+|(?<=!)\s+|(?<=\?)\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(group).strip())
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators:
            # No separators left, force-split by chunk_size
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if separator == "":
            # Character-level split
            chunks = []
            for i in range(0, len(current_text), self.chunk_size):
                chunks.append(current_text[i : i + self.chunk_size])
            return chunks

        parts = current_text.split(separator)

        # Merge small parts back together, then recursively split large ones
        chunks: list[str] = []
        current_chunk = ""
        for part in parts:
            candidate = current_chunk + separator + part if current_chunk else part
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        # Recursively split any chunks that are still too large
        result: list[str] = []
        for c in chunks:
            if len(c) <= self.chunk_size:
                result.append(c)
            else:
                result.extend(self._split(c, next_separators))

        return result


class MarkdownHeaderChunker:
    """
    Custom chunker that splits markdown text by headers, preserving section context.

    Strategy:
        1. Split text at markdown headers (# ## ### etc.)
        2. Prepend parent header(s) to each chunk for context
        3. If a section is too large, fall back to RecursiveChunker within that section
        4. Merge very small sections with the next section

    This is ideal for technical documentation where headers define topic boundaries.
    """

    def __init__(self, chunk_size: int = 500, min_chunk_size: int = 50) -> None:
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split by markdown headers, keeping the header with its content
        sections = re.split(r'(?=^#{1,4}\s)', text, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]

        if not sections:
            return [text]

        # Track parent headers for context
        chunks: list[str] = []
        header_stack: list[str] = []  # (level, header_text)

        for section in sections:
            # Detect header level
            header_match = re.match(r'^(#{1,4})\s+(.+)', section)
            if header_match:
                level = len(header_match.group(1))
                # Pop headers at same or deeper level
                header_stack = [(lvl, h) for lvl, h in header_stack if lvl < level]
                header_stack.append((level, header_match.group(0).strip()))

            # Build context prefix from parent headers
            if header_match and len(header_stack) > 1:
                context_prefix = " > ".join(h for _, h in header_stack[:-1]) + "\n"
            else:
                context_prefix = ""

            full_chunk = context_prefix + section

            if len(full_chunk) <= self.chunk_size:
                chunks.append(full_chunk)
            else:
                # Fall back to recursive splitting for large sections
                sub_chunker = RecursiveChunker(chunk_size=self.chunk_size)
                sub_chunks = sub_chunker.chunk(section)
                for sc in sub_chunks:
                    chunks.append((context_prefix + sc) if context_prefix else sc)

        # Merge very small chunks with the next one
        merged: list[str] = []
        for c in chunks:
            if merged and len(merged[-1]) < self.min_chunk_size:
                merged[-1] = merged[-1] + "\n\n" + c
            else:
                merged.append(c)

        return merged


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    mag_a = math.sqrt(_dot(vec_a, vec_a))
    mag_b = math.sqrt(_dot(vec_b, vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunks = FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(text)
        sentence_chunks = SentenceChunker(max_sentences_per_chunk=3).chunk(text)
        recursive_chunks = RecursiveChunker(chunk_size=chunk_size).chunk(text)

        def stats(chunks):
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0
            return {"count": count, "avg_length": avg_length, "chunks": chunks}

        return {
            "fixed_size": stats(fixed_chunks),
            "by_sentences": stats(sentence_chunks),
            "recursive": stats(recursive_chunks),
        }
