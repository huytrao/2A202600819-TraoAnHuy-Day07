from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import re


@dataclass
class Document:
    """
    A simple document/chunk object.

    content: text content of the document/chunk
    metadata: extra information such as doc_id, source, category, topic
    """
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class FixedSizeChunker:
    """
    Split text into fixed-size character chunks.

    Example:
        chunk_size = 500
        overlap = 50

    Chunk 1: characters 0 -> 500
    Chunk 2: characters 450 -> 950
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0:
            raise ValueError("overlap must be non-negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if text is None:
            return []

        text = text.strip()

        if not text:
            return []

        chunks = []
        step = self.chunk_size - self.overlap

        for start in range(0, len(text), step):
            end = start + self.chunk_size
            piece = text[start:end].strip()

            if piece:
                chunks.append(piece)

            if end >= len(text):
                break

        return chunks


class SentenceChunker:
    """
    Split text by sentence boundaries, then group sentences into chunks.

    This is useful when we want to avoid cutting in the middle of a sentence.
    """

    def __init__(
        self,
        max_sentences_per_chunk: int = 2,
        chunk_size: int = 500,
        overlap: int = 50,
    ):
        if max_sentences_per_chunk <= 0:
            raise ValueError("max_sentences_per_chunk must be positive")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0:
            raise ValueError("overlap must be non-negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.fallback_chunker = FixedSizeChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def _split_sentences(self, text: str) -> list[str]:
        """
        Split text into sentences using punctuation.

        Supports:
        - English punctuation: . ? !
        - Some Asian punctuation: 。 ？ ！
        """
        text = re.sub(r"\s+", " ", text.strip())

        if not text:
            return []

        sentences = re.split(r"(?<=[.!?。！？])\s+", text)

        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def chunk(self, text: str) -> list[str]:
        if text is None:
            return []

        text = text.strip()

        if not text:
            return []

        sentences = self._split_sentences(text)

        if not sentences:
            return []

        # If text has no clear sentence boundary and is too long,
        # fallback to fixed-size chunking.
        if len(sentences) == 1 and len(sentences[0]) > self.chunk_size:
            return self.fallback_chunker.chunk(text)

        chunks = []
        current_sentences: list[str] = []

        def flush_current() -> None:
            if current_sentences:
                chunks.append(" ".join(current_sentences).strip())
                current_sentences.clear()

        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                flush_current()
                chunks.extend(self.fallback_chunker.chunk(sentence))
                continue

            if not current_sentences:
                current_sentences.append(sentence)
                continue

            candidate_sentences = current_sentences + [sentence]
            candidate = " ".join(candidate_sentences)

            if (
                len(candidate_sentences) <= self.max_sentences_per_chunk
                and len(candidate) <= self.chunk_size
            ):
                current_sentences = candidate_sentences
                continue

            flush_current()
            current_sentences.append(sentence)

        flush_current()
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators from large structure to small structure.

    This version is designed for your preference:
    - NO overlap by default.
    - No broken text such as "his helps retrieval."
    - Keeps headings and related content together when possible.

    Example separators:
        ["\\n## ", "\\n### ", "\\n\\n", ". ", "? ", "! ", " ", ""]

    Meaning:
        1. Try splitting by markdown heading.
        2. Then by paragraph.
        3. Then by sentence.
        4. Then by word.
        5. Finally hard split by character length.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 0,
        separators: list[str] | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0:
            raise ValueError("overlap must be non-negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

        self.separators = separators or [
            "\n## ",
            "\n### ",
            "\n\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ]

    def chunk(self, text: str) -> list[str]:
        """
        Main public method.

        Input:
            text: original document text

        Output:
            list of chunks
        """
        if text is None:
            return []

        text = text.strip()

        if not text:
            return []

        pieces = self._split(text, self.separators)
        chunks = self._merge_pieces(pieces)

        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text using the provided separators.

        Base cases:
        - If text length <= chunk_size, return it.
        - If no separators remain, hard split by character length.
        """
        text = text.strip()

        if not text:
            return []

        # Base case 1: already small enough
        if len(text) <= self.chunk_size:
            return [text]

        # Base case 2: no separators left
        if not separators:
            return self._hard_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Final fallback separator
        if separator == "":
            return self._hard_split(text)

        parts = text.split(separator)

        # If this separator does not split the text,
        # try the next smaller separator.
        if len(parts) == 1:
            return self._split(text, remaining_separators)

        result = []

        for i, part in enumerate(parts):
            part = part.strip()

            if not part:
                continue

            # Add separator back so heading/punctuation is not lost.
            # Example:
            # text.split("\n## ") removes "\n## "
            # so we add "## " back for later parts.
            if i > 0:
                clean_sep = separator.strip()

                if clean_sep:
                    part = clean_sep + " " + part

            if len(part) <= self.chunk_size:
                result.append(part.strip())
            else:
                # Still too large, split again using smaller separators.
                result.extend(self._split(part, remaining_separators))

        return result

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """
        Merge small pieces into chunks up to chunk_size.

        Important:
        - This version does NOT use overlap.
        - It does not copy characters from the previous chunk.
        - This avoids ugly broken text like "his helps retrieval."
        """
        chunks = []
        current = ""

        for piece in pieces:
            piece = piece.strip()

            if not piece:
                continue

            if not current:
                current = piece
                continue

            candidate = current + "\n\n" + piece

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                chunks.append(current.strip())
                current = piece

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """
        Final fallback: split text by character length.

        No overlap.
        """
        text = text.strip()

        if not text:
            return []

        chunks = []

        for start in range(0, len(text), self.chunk_size):
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

        return chunks


def _dot(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute the dot product between two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length")

    return sum(a * b for a, b in zip(vec_a, vec_b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (norm(a) * norm(b))

    If one vector has zero magnitude, return 0.0.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length")

    dot_product = _dot(vec_a, vec_b)

    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """
    Compare FixedSizeChunker, SentenceChunker, and RecursiveChunker.

    Output includes:
    - count
    - avg_length
    - min_length
    - max_length
    - chunks
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 0):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0:
            raise ValueError("overlap must be non-negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def compare(self, text: str, chunk_size: int = None, overlap: int = None) -> dict[str, dict[str, Any]]:
        if chunk_size is None:
            chunk_size = self.chunk_size
        if overlap is None:
            overlap = self.overlap

        strategies = {
            "fixed_size": FixedSizeChunker(
                chunk_size=chunk_size,
                overlap=overlap,
            ),
            "by_sentences": SentenceChunker(
                max_sentences_per_chunk=2,
                chunk_size=chunk_size,
                overlap=overlap,
            ),
            "recursive": RecursiveChunker(
                chunk_size=chunk_size,
                overlap=0,
            ),
        }

        results = {}
        for strategy_name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            results[strategy_name] = self._compute_stats(chunks)

        return results

    def _compute_stats(self, chunks: list[str]) -> dict[str, Any]:
        if not chunks:
            return {
                "chunk_count": 0,
                "avg_length": 0,
                "min_length": 0,
                "max_length": 0,
                "chunks": [],
            }

        lengths = [len(chunk) for chunk in chunks]

        return {
            "count": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "chunks": chunks,
        }


if __name__ == "__main__":
    text = """
## Introduction

Retrieval-Augmented Generation helps LLMs answer using external knowledge.

## Chunking

Chunking splits long documents into smaller pieces. This helps retrieval.

## Vector Store

A vector store saves embeddings and supports semantic search.
"""

    chunker = RecursiveChunker(chunk_size=120, overlap=0)
    chunks = chunker.chunk(text)

    for i, chunk in enumerate(chunks, start=1):
        print("---- chunk", i)
        print(chunk)
        print("length:", len(chunk))