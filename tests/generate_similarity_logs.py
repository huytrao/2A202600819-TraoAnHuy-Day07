import os
import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để có thể import src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.chunking import compute_similarity

def generate_similarity_logs():
    log_dir = project_root / "tests" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "similarity_results.log"

    # Lấy hàm embedding, ưu tiên dùng LocalEmbedder để có kết quả thực tế
    try:
        from src.embeddings import LocalEmbedder
        embedder = LocalEmbedder()
        print("Sử dụng LocalEmbedder để tạo vector embeddings...")
    except Exception:
        from src.embeddings import _mock_embed
        embedder = _mock_embed
        print("Sử dụng mock_embedder vì không load được LocalEmbedder...")
        
    # Chuẩn bị 5 cặp câu có độ liên quan đa dạng
    sentence_pairs = [
        # Pair 1: Rất giống nhau về ý nghĩa và từ vựng
        (
            "Retrieval-Augmented Generation helps LLMs access external knowledge.",
            "RAG systems allow language models to retrieve outside information."
        ),
        # Pair 2: Cùng chủ đề nhưng khác trọng tâm
        (
            "Vector databases store embeddings for similarity search.",
            "Chunking text into smaller pieces is crucial for good RAG performance."
        ),
        # Pair 3: Hoàn toàn không liên quan
        (
            "Retrieval-Augmented Generation helps LLMs access external knowledge.",
            "I love eating pizza with extra cheese on weekends."
        ),
        # Pair 4: Trái ngược/đối lập (đôi khi embedding vẫn cho similarity khá cao vì cùng context)
        (
            "Python is a great language for beginners.",
            "Python is a terrible language for beginners."
        ),
        # Pair 5: Tiếng Việt - Cùng ý nghĩa
        (
            "Tôi thích uống cà phê đen đá không đường.",
            "Ly cà phê đen đá không đường là thức uống yêu thích của tôi."
        )
    ]
    
    with open(log_file, "w", encoding="utf-8") as log:
        log.write("=== COSINE SIMILARITY RESULTS (Hỗ trợ Exercise 3.3) ===\n")
        log.write("Bài tập yêu cầu dự đoán cặp nào có độ tương đồng cao nhất/thấp nhất.\n")
        log.write("Sau đây là kết quả thực tế để bạn đối chiếu với dự đoán của mình:\n\n")
        
        for idx, (sent1, sent2) in enumerate(sentence_pairs, 1):
            # Lấy vector cho từng câu (cần chuyển list qua cho hàm compute_similarity nếu hàm yêu cầu)
            vec1 = embedder(sent1)
            vec2 = embedder(sent2)
            
            # Tính toán similarity
            similarity = compute_similarity(vec1, vec2)
            
            log.write(f"Pair {idx}:\n")
            log.write(f"  Sentence A: {sent1}\n")
            log.write(f"  Sentence B: {sent2}\n")
            log.write(f"  --> Actual Cosine Similarity: {similarity:.4f}\n")
            log.write("-" * 50 + "\n")
            
    print(f"\nĐã ghi log kết quả similarity vào: {log_file}")

if __name__ == "__main__":
    generate_similarity_logs()
