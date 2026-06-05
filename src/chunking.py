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
        
        parts = re.split(r'(\. |\! |\? |\.\n)', text)
        sentences = []
        for part in parts:
            if part in {". ", "! ", "? ", ".\n"}:
                if sentences:
                    sentences[-1] += part
                else:
                    sentences.append(part)
            else:
                sentences.append(part)
                
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunks.append(" ".join(sentences[i : i + self.max_sentences_per_chunk]))
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

        separator = ""
        next_separators = []
        
        for i, sep in enumerate(remaining_separators):
            if sep == "" or sep in current_text:
                separator = sep
                next_separators = remaining_separators[i + 1:]
                break
        
        if separator == "":
            parts = list(current_text)
        else:
            parts = current_text.split(separator)
            
        chunks = []
        current_chunk = ""
        
        for i, part in enumerate(parts):
            part_text = part + separator if i < len(parts) - 1 and separator != "" else part
            
            if len(current_chunk) + len(part_text) <= self.chunk_size:
                current_chunk += part_text
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(part_text) > self.chunk_size:
                    sub_chunks = self._split(part_text, next_separators)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = part_text
                    
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed = FixedSizeChunker(chunk_size=chunk_size).chunk(text)
        sentence = SentenceChunker().chunk(text)
        recursive = RecursiveChunker(chunk_size=chunk_size).chunk(text)
        
        return {
            "fixed_size": {
                "count": len(fixed),
                "avg_length": sum(len(c) for c in fixed) / len(fixed) if fixed else 0.0,
                "chunks": fixed,
            },
            "by_sentences": {
                "count": len(sentence),
                "avg_length": sum(len(c) for c in sentence) / len(sentence) if sentence else 0.0,
                "chunks": sentence,
            },
            "recursive": {
                "count": len(recursive),
                "avg_length": sum(len(c) for c in recursive) / len(recursive) if recursive else 0.0,
                "chunks": recursive,
            }
        }
