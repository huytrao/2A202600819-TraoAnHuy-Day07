from __future__ import annotations

import os
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from src.chunking import CustomChunkerbyMarkdownHeadings as CustomChunker

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/metadata.json",
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt", ".json"}
    documents: list[Document] = []
    markdown_chunker = CustomChunker(chunk_size=250, overlap=50)
    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: {', '.join(allowed_extensions)})")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        
        if path.suffix.lower() == ".json":
            try:
                data = json.loads(content)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    content_body = item.get("content")
                    if not content_body and "file_name" in item:
                        target_file = Path("data") / item.get("file_name")
                        if target_file.exists():
                            content_body = target_file.read_text(encoding="utf-8")
                    
                    if content_body:
                        print(f"DEBUG: Nạp doc {item.get('id')} với độ dài content: {len(content_body)} ký tự")
                        # Cần chia nhỏ nội dung nạp từ JSON để kết quả search chính xác hơn
                        chunks = markdown_chunker.chunk(content_body)
                        for i, chunk_text in enumerate(chunks):
                            documents.append(
                                Document(
                                    id=f"{item.get('id', item.get('file_name', path.stem))}_p{i}",
                                    content=chunk_text,
                                    metadata=item.get("metadata", {"source": str(path)})
                                )
                            )
            except Exception as e:
                print(f"Error parsing JSON file {path}: {e}")
        else:
            chunks = markdown_chunker.chunk(content)
            for i, chunk_text in enumerate(chunks):
              documents.append(
                Document(
                        id=f"{path.stem}_part{i}", 
                        content=chunk_text,
                        metadata={
                            "source": str(path), 
                            "chunk_index": i,
                            "extension": path.suffix.lower()
                        },
                    )
            )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Tóm tắt thông tin quan trọng từ các tài liệu đã nạp."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt, .json")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print(" python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    store = EmbeddingStore(collection_name="manual_test_store", embedding_fn=embedder)
    store.add_documents(docs)

    print(f"\nStored {store.get_collection_size()} documents in EmbeddingStore")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source', 'N/A')}")
        print(f"   content preview: {result['content'][:500].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(f"Question: {query}")
    print("Agent answer:")
    print(agent.answer(query, top_k=3))
    return 0

def run_retrieval_benchmark(benchmark_file: str):
    """Khởi tạo môi trường và chạy kiểm thử truy xuất tự động."""
    # 1. Thiết lập Embedder
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    # 2. Nạp tài liệu và khởi tạo Vector Store
    docs = load_documents_from_files(SAMPLE_FILES)
    if not docs:
        print("❌ Không có tài liệu nào được nạp để chạy benchmark.")
        return 1

    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=embedder)
    store.add_documents(docs)
    print(f"✅ Đã nạp {store.get_collection_size()} tài liệu vào Store cho benchmark.")

    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    for item in benchmarks:
        query = item['query']
        expected = item['expected_source']
        meta_filter = item.get('metadata_filter')
        # Hỗ trợ lọc theo metadata nếu benchmark yêu cầu
        if meta_filter:
            results = store.search_with_filter(query, top_k=3, metadata_filter=meta_filter)
        else:
            results = store.search(query, top_k=3)
        
        print(f"\nQuery: {query}")
        print(f"Expected: {expected}")
        
        # Kiểm tra xem file mong đợi có nằm trong top 3 không
        found = False
        for i, res in enumerate(results):
            source = res['metadata'].get('source', '')
            doc_id = res.get('id', '')
            print(f"  {i+1}. Found: {source} (ID: {doc_id}, Score: {res['score']:.3f})")
            
            # So khớp linh hoạt giữa ID tài liệu và tên file mong đợi
            if expected in source or expected in doc_id or doc_id in expected:
                found = True
        
        if found:
            print(">>> ✅ PASS: Tìm đúng nguồn")
        else:
            print(">>> ❌ FAIL: Không tìm thấy nguồn mong đợi")
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_retrieval_benchmark(benchmark_file="benchmark/benchmark.json")


if __name__ == "__main__":
    raise SystemExit(main())
