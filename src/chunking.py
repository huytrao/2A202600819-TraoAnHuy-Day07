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
        if not text or not text.strip():
            return []

        # Sử dụng biểu thức chính quy để phân tách câu theo đúng mô tả: ". ", "! ", "? ", ".\n"
        # Sử dụng dấu ngoặc đơn để bắt cả ký tự phân tách nếu cần giữ lại, hoặc split trực tiếp:
        sentences = re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks: list[str] = []
        # Gom nhóm các câu theo số lượng max_sentences_per_chunk (không chồng lấp theo mô tả của class này)
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            group = sentences[i : i + self.max_sentences_per_chunk]
            chunk_text = " ".join(group).strip()
            if chunk_text:
                chunks.append(chunk_text)

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
        if not text or not text.strip():
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Nếu đoạn text đã đủ nhỏ hơn kích thước quy định, giữ nguyên và trả về
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Nếu đã thử hết tất cả ký tự phân tách mà vẫn quá dài, cắt cứng theo ký tự kí tự trống ""
        if not remaining_separators:
            return [current_text[i : i + self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Thực hiện phân tách văn bản bằng ký tự hiện tại
        if separator == "":
            splits = list(current_text)
        else:
            splits = current_text.split(separator)

        chunks: list[str] = []
        current_chunk = ""

        for part in splits:
            # Nếu bản thân phần tử con sau tách đã vượt quá kích thước, đệ quy sâu xuống với separator kế tiếp
            if len(part) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.extend(self._split(part, next_separators))
            else:
                # Kiểm tra xem nếu gộp thêm phần tử này vào đoạn hiện tại có bị quá tải kích thước không
                separator_to_join = separator if current_chunk else ""
                if len(current_chunk) + len(separator_to_join) + len(part) <= self.chunk_size:
                    current_chunk += separator_to_join + part
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        return [c for c in chunks if c.strip() or c == ""]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = _dot(vec_a, vec_b)
    
    # Tính Magnitude (độ dài) của từng vector
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))

    # Tránh lỗi chia cho 0 nếu một trong hai vector rỗng/toàn số 0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)

class CustomChunkerbyMarkdownHeadings:
    """
    Split Markdown text based on Heading levels (#, ##, ###).
    Ensures that content under a heading is kept together.
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 0) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        
        # Split by heading markers (#, ##, ...) while keeping the marker
        pattern = r'(?=^#{1,6}\s)'
        segments = re.split(pattern, text, flags=re.MULTILINE)
        
        chunks: list[str] = []
        current_chunk = ""

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
                
            # If a single segment is massive, we append it as-is or force split it
            if len(segment) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                chunks.append(segment)
            elif len(current_chunk) + len(segment) + 1 <= self.chunk_size:
                current_chunk += ("\n" if current_chunk else "") + segment
            else:
                chunks.append(current_chunk)
                current_chunk = segment
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return [c for c in chunks if c.strip()]


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        """
        Khởi tạo các chiến lược chunking, chạy thử nghiệm trên văn bản đầu vào 
        và tổng hợp các chỉ số thống kê cơ bản để so sánh hiệu năng.
        """
        # Khởi tạo 3 chiến lược chunker
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=int(chunk_size * 0.1))
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)
        custom_chunker = CustomChunkerbyMarkdownHeadings(chunk_size=chunk_size, overlap=int(chunk_size * 0.5))

        # Chạy chunk văn bản dữ liệu đầu vào
        fixed_chunks = fixed_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)
        custom_chunks = custom_chunker.chunk(text)

        # Định dạng cấu trúc phẳng (không nhóm vào dict 'stats') và đổi tên key sang 'by_sentences'
        return {
            "fixed_size": {
                "count": len(fixed_chunks),
                "avg_length": sum(len(c) for c in fixed_chunks) / len(fixed_chunks) if fixed_chunks else 0.0,
                "chunks": fixed_chunks
            },
            "by_sentences": {  # Đổi tên key từ 'sentence' thành 'by_sentences'
                "count": len(sentence_chunks),
                "avg_length": sum(len(c) for c in sentence_chunks) / len(sentence_chunks) if sentence_chunks else 0.0,
                "chunks": sentence_chunks
            },
            "recursive": {
                "count": len(recursive_chunks),
                "avg_length": sum(len(c) for c in recursive_chunks) / len(recursive_chunks) if recursive_chunks else 0.0,
                "chunks": recursive_chunks
            },
            "by_markdown_headings": {
                "count": len(custom_chunks),
                "avg_length": sum(len(c) for c in custom_chunks) / len(custom_chunks) if custom_chunks else 0.0,
                "chunks": custom_chunks
            }
        }