import os
import sys
import json
from pathlib import Path

# Thêm thư mục gốc vào sys.path để có thể import src
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.chunking import ChunkingStrategyComparator

def generate_logs():
    data_dir = project_root / "data"
    log_dir = project_root / "tests" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "chunking_results.log"
    metadata_file = data_dir / "metadata.json"
    
    comparator = ChunkingStrategyComparator()
    
    # Lấy danh sách file từ metadata.json
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)
        
    with open(log_file, "w", encoding="utf-8") as f:
        for item in metadata_list:
            file_name = item.get("file_name")
            if not file_name:
                continue
                
            file_path = data_dir / file_name
            if not file_path.exists():
                print(f"Warning: {file_path} không tồn tại.")
                continue
                
            with open(file_path, "r", encoding="utf-8") as rf:
                text = rf.read()
                
            f.write(f"=== Chunking Results for {file_name} ===\n")
            
            # Thực thi chunking comparison
            results = comparator.compare(text, chunk_size=300)
            
            for strategy, stats in results.items():
                f.write(f"Strategy: {strategy}\n")
                f.write(f"  - Count: {stats['count']}\n")
                f.write(f"  - Average Length: {stats['avg_length']:.2f}\n")
                f.write(f"  - Sample Chunks:\n")
                
                # Lấy tối đa 3 chunk mẫu
                sample_chunks = stats["chunks"][:3]
                for i, chunk in enumerate(sample_chunks, 1):
                    # Hiển thị rút gọn nếu dài quá
                    display_chunk = chunk.replace("\n", " ")
                    if len(display_chunk) > 100:
                        display_chunk = display_chunk[:97] + "..."
                    f.write(f"      [{i}] {display_chunk}\n")
            
            f.write("\n==================================================\n\n")
            
    print(f"Đã ghi log thành công vào {log_file}")

if __name__ == "__main__":
    generate_logs()
