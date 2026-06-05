from __future__ import annotations

import math
import re


# from langchain_text_splitters import SemanticChunker as LangchainSemanticChunker
# from langchain_huggingface import HuggingFaceEmbeddings


class SemanticChunker:
    """Semantic chunking strategy based on sentence embeddings using all-MiniLM-L6-v2."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", threshold_percentile: int = 95):
        # Khởi tạo embedding và splitter trực tiếp, nếu thiếu thư viện sẽ raise ImportError ngay lập tức
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = LangchainSemanticChunker(
            self.embeddings, 
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=threshold_percentile
        )
    def chunk(self, text: str) -> list[str]:
        if not text.strip():
            return []
        documents = self.text_splitter.create_documents([text])
        chunks = [doc.page_content for doc in documents]
        
        # Fix: Đảm bảo nếu Langchain ko cắt được (vd text quá ngắn) thì vẫn trả về ít nhất 1 chunk
        if not chunks and text.strip():
            return [text.strip()]
            
        return chunks
        
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
        if not text.strip():
            return []

        # Sử dụng regex với Lookbehind (?<=...) để giữ lại các ký tự kết thúc câu (.!? và \n)
        # Kết hợp bắt cặp dấu cách hoặc xuống dòng đi liền sau.
        sentence_endings = r"(?<=[.!?])\s+|(?<=\.\n)"
        
        # Tách chuỗi và lọc bỏ các chuỗi rỗng sau khi strip
        sentences = [s.strip() for s in re.split(sentence_endings, text) if s.strip()]
        
        chunks: list[str] = []
        # Nhóm các câu lại theo số lượng tối đa cấu hình
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_sentences = sentences[i : i + self.max_sentences_per_chunk]
            chunks.append(" ".join(chunk_sentences))
            
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
        # Nếu đoạn text đã nhỏ hơn chunk_size, không cần chia nhỏ thêm nữa
        if len(current_text) <= self.chunk_size:
            return [current_text]

        # Nếu không còn separator nào để chia, bắt buộc cắt ép buộc theo chunk_size
        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        # Lấy separator hiện tại có độ ưu tiên cao nhất
        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        # Tách chuỗi dựa trên separator hiện tại
        if separator == "":
            # Phân tách từng ký tự
            splits = list(current_text)
        else:
            # Dùng re.escape đề phòng separator chứa ký tự đặc biệt của regex
            splits = current_text.split(separator)

        final_chunks: list[str] = []
        good_splits: list[str] = []

        for s in splits:
            # Nếu s quá dài, cần mang đi đệ quy với separator tiếp theo
            if len(s) > self.chunk_size:
                # Nếu trước đó đã có một số đoạn gom lại đủ dùng, đẩy vào final_chunks trước
                if good_splits:
                    final_chunks.append(separator.join(good_splits))
                    good_splits = []
                
                # Gọi đệ quy cho đoạn text quá cỡ này
                recursive_chunks = self._split(s, next_separators)
                final_chunks.extend(recursive_chunks)
            else:
                # Kiểm tra xem nếu gộp thêm `s` vào nhóm hiện tại có bị quá kích thước không
                # Cần cộng thêm độ dài của separator nối giữa chúng
                current_length = sum(len(x) for x in good_splits) + len(separator) * max(0, len(good_splits) - 1)
                potential_length = current_length + len(separator) + len(s) if good_splits else len(s)

                if potential_length <= self.chunk_size:
                    good_splits.append(s)
                else:
                    # Nếu vượt quá, đóng gói nhóm cũ lại và mở nhóm mới với `s`
                    if good_splits:
                        final_chunks.append(separator.join(good_splits))
                    good_splits = [s]

        # Đóng gói phần text còn dư lại ở cuối vòng lặp
        if good_splits:
            final_chunks.append(separator.join(good_splits))

        return final_chunks


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
    
    # Tính L2 norm (magnitude) của từng vector
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))

    # Tránh lỗi DivisionByZero
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        # Khởi tạo các bộ chunkers với chunk_size truyền vào cấu hình tương ứng
        # (Đối với SentenceChunker, giả định ước lượng 1 câu ~ 50-70 ký tự, lấy max là 3 câu)
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=20)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)
        # semantic_chunker = SemanticChunker(threshold_percentile=95)

        # Tiến hành chunking dữ liệu đầu vào
        fixed_chunks = fixed_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)
        # semantic_chunks = semantic_chunker.chunk(text)

        # Hàm helper nhanh giúp tính toán thống kê cơ bản
        def _get_stats(chunks: list[str]) -> dict:
            if not chunks:
                return {"count": 0, "avg_length": 0.0, "chunks": []}
            lengths = [len(c) for c in chunks]
            return {
                "count": len(chunks),
                "avg_length": sum(lengths) / len(lengths),
                "chunks": chunks
            }

        return {
            "fixed_size": _get_stats(fixed_chunks),
            "by_sentences": _get_stats(sentence_chunks),
            "recursive": _get_stats(recursive_chunks),
            # "semantic": _get_stats(semantic_chunks)
        }