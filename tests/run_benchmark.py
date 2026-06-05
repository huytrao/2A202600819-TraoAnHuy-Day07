import os
import sys
import json
from pathlib import Path

# Thêm thư mục gốc vào sys.path để có thể import src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models import Document
from src.store import EmbeddingStore
from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker

def run_benchmark():
    benchmark_file = project_root / "benchmark" / "benchmark.json"
    data_dir = project_root / "data"
    log_dir = project_root / "tests" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "benchmark_results.log"
    metadata_file = data_dir / "metadata.json"

    # 1. Đọc danh sách file từ metadata.json
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
        
    # 2. Chuẩn bị các chiến lược chunking
    chunking_strategies = {
        "fixed_size": FixedSizeChunker(chunk_size=300, overlap=50),
        "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
        "recursive": RecursiveChunker(chunk_size=300)
    }
    
    # Chuẩn bị embedder dùng chung
    try:
        from src.embeddings import LocalEmbedder
        embedder = LocalEmbedder()
    except Exception:
        from src.embeddings import _mock_embed
        embedder = _mock_embed

    # Đọc file benchmark
    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark_queries = json.load(f)

    with open(log_file, "w", encoding="utf-8") as log:
        log.write("=== BENCHMARK RETRIEVAL RESULTS ===\n")
        log.write("Thông tin này hỗ trợ Exercise 3.2 (Bảng Benchmark Queries)\n\n")
        
        for strategy_name, chunker in chunking_strategies.items():
            log.write(f"*** EVALUATING STRATEGY: {strategy_name} ***\n\n")
            
            # Nạp và cắt nhỏ (chunk) nội dung các documents theo chiến lược
            docs = []
            for item in metadata_list:
                file_name = item.get("file_name")
                if not file_name:
                    continue
                    
                file_path = data_dir / file_name
                if not file_path.exists():
                    continue
                    
                with open(file_path, "r", encoding="utf-8") as rf:
                    text = rf.read()
                    
                chunks = chunker.chunk(text)
                for i, chunk_text in enumerate(chunks):
                    docs.append(Document(
                        id=f"{file_name}_chunk_{i}",
                        content=chunk_text,
                        metadata={"source": file_name, "category": item.get("metadata", {}).get("category", "")}
                    ))
            
            # Khởi tạo Vector Store và nạp chunks cho chiến lược này
            store = EmbeddingStore(collection_name=f"benchmark_store_{strategy_name}", embedding_fn=embedder)
            store.add_documents(docs)

            correct_retrievals = 0
            total_queries = len(benchmark_queries)

            for q in benchmark_queries:
                query_text = q["query"]
                gold_answer = q.get("gold_answer", "N/A")
                expected_source = q["expected_source"]
                metadata_filter = q.get("metadata_filter", {})
                
                # Nếu có filter trong benchmark thì áp dụng, nếu rỗng thì pass None
                filter_arg = metadata_filter if metadata_filter else None
                
                # Tìm kiếm top 3 kết quả
                if hasattr(store, 'search_with_filter'):
                    results = store.search_with_filter(query_text, top_k=3, metadata_filter=filter_arg)
                else:
                    results = store.search(query_text, top_k=3)
                
                # Lấy danh sách source trả về (loại bỏ trùng lặp)
                retrieved_sources = list(set([r["metadata"].get("source", "") for r in results]))
                
                # Kiểm tra xem file nguồn dự kiến có nằm trong top kết quả trả về không
                is_correct = False
                for r_src in retrieved_sources:
                    if r_src in expected_source or expected_source in r_src:
                        is_correct = True
                        break
                        
                if is_correct:
                    correct_retrievals += 1
                    
                log.write(f"Query ID: {q['id']}\n")
                log.write(f"Question: {query_text}\n")
                log.write(f"Gold Answer: {gold_answer}\n")
                log.write(f"Expected Source: {expected_source}\n")
                log.write(f"Retrieved Sources: {retrieved_sources}\n")
                log.write(f"Result: {'PASS' if is_correct else 'FAIL'}\n")
                
                log.write("Top 3 Chunks Retrieved:\n")
                for idx, r in enumerate(results, 1):
                    # In ra đoạn trích của chunk để dễ nhận diện trong Exercise 3.2
                    chunk_preview = r.get("content", "").replace("\n", " ")
                    if len(chunk_preview) > 150:
                        chunk_preview = chunk_preview[:147] + "..."
                    log.write(f"  [{idx}] ID: {r.get('id')} | Score: {r.get('score', 0):.3f}\n")
                    log.write(f"      Content: {chunk_preview}\n")
                
                log.write("-" * 50 + "\n")

            accuracy = (correct_retrievals / total_queries) * 100 if total_queries > 0 else 0
            log.write(f"\n[SUMMARY - {strategy_name}]\n")
            log.write(f"Total Queries: {total_queries}\n")
            log.write(f"Correct Retrievals: {correct_retrievals}\n")
            log.write(f"Retrieval Accuracy (Top-3): {accuracy:.2f}%\n")
            log.write("=" * 60 + "\n\n")
        
    print(f"Đã hoàn thành chạy benchmark cho 3 chiến lược chunking. Chi tiết xem tại: {log_file}")

if __name__ == "__main__":
    run_benchmark()
