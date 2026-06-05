from __future__ import annotations

import math
from typing import Any, Callable

# Import an toàn: Chỉ dùng cho việc tính toán fallback
from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document

class EmbeddingStore:
    """
    A vector store for text chunks.
    Tries to use ChromaDB if available; falls back to an in-memory store.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        
        # Tạo một chuỗi duy nhất cho mỗi instance để cô lập bộ nhớ Chroma giữa các test
        import uuid
        unique_suffix = uuid.uuid4().hex[:8]
        self._collection_name = f"{collection_name}_{unique_suffix}"
        
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0
        self._use_chroma = False

        try:
            import chromadb
            client = chromadb.EphemeralClient()
            
            self._collection = client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except (ImportError, Exception):
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        
        # Lấy metadata hiện có
        metadata = doc.metadata.copy() if doc.metadata else {}
        
        # QUAN TRỌNG: ChromaDB không nhận metadata rỗng. 
        # Nếu rỗng, hãy thêm một key mặc định để pass qua validator của Chroma.
        if not metadata:
            metadata = {"source": "unknown"}
            
        record = {
            "id": getattr(doc, "id", None) or f"id_{self._next_index}",
            "content": doc.content,
            "embedding": embedding,
            "metadata": metadata
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Hàm tìm kiếm trên RAM, đảm bảo luôn trả về trường 'score'."""
        if not records:
            return []
            
        query_embedding = self._embedding_fn(query)
        scored_records = []
        
        for record in records:
            # Tính toán độ tương đồng
            score = compute_similarity(query_embedding, record["embedding"])
            scored_records.append({
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": float(score)
            })
            
        # Sắp xếp theo score giảm dần
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, documents: list[Document]) -> None:
        """
        Trích xuất thông tin từ danh sách Document, tính toán vector embedding 
        và lưu trữ đồng bộ vào bộ nhớ RAM cục bộ hoặc ChromaDB collection.
        """
        import uuid

        if self._use_chroma and self._collection is not None:
            ids = []
            documents_content = []
            embeddings = []
            metadatas = []

            # Lấy danh sách tất cả các ID hiện tại đang có trong ChromaDB để kiểm tra trùng
            existing_ids = set()
            try:
                current_data = self._collection.get(include=[])
                if current_data and "ids" in current_data:
                    existing_ids = set(current_data["ids"])
            except Exception:
                pass

            for doc in documents:
                record = self._make_record(doc)
                final_id = record["id"]
                
                # QUAN TRỌNG: Nếu ID này đã tồn tại trong database từ lần add trước, 
                # hãy đổi nó thành một ID mới duy nhất để ChromaDB ghi nhận thêm bản ghi mới thay vì đè lên bản ghi cũ.
                if final_id in existing_ids:
                    final_id = f"{final_id}_{uuid.uuid4().hex[:4]}"
                
                # Cập nhật lại ID mới cho cả bản ghi và danh sách kiểm tra nhanh
                existing_ids.add(final_id)
                
                ids.append(final_id)
                documents_content.append(record["content"])
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"])

            if ids:
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents_content,
                    metadatas=metadatas
                )
        else:
            # Logic lưu trữ trên RAM cục bộ (Môi trường Fallback)
            for doc in documents:
                record = self._make_record(doc)
                
                # Xử lý trùng ID cho lưu trữ RAM cục bộ
                existing_ids = [r["id"] for r in self._store]
                if record["id"] in existing_ids:
                    record["id"] = f"{record['id']}_{uuid.uuid4().hex[:4]}"
                    
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Tìm kiếm top_k tài liệu liên quan nhất."""
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "embeddings"]
            )
            
            formatted_results = []
            if results and results.get("ids") and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    # Kiểm tra mảng NumPy an toàn bằng cách dùng 'is not None' thay vì xét điều kiện 'if doc_emb' trực tiếp
                    doc_emb = results["embeddings"][0][i] if results.get("embeddings") else None
                    
                    if doc_emb is not None:
                        # Nếu doc_emb là mảng numpy hoặc list, ép kiểu hoặc tính toán an toàn
                        if hasattr(doc_emb, "tolist"):
                            score = compute_similarity(query_embedding, doc_emb.tolist())
                        else:
                            score = compute_similarity(query_embedding, doc_emb)
                    else:
                        score = 1.0
                    
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": float(score)
                    })
            return formatted_results
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Trả về tổng số bản ghi."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict[str, Any]]:
        """Tìm kiếm kết hợp lọc metadata."""
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            # Chuẩn hóa filter cho ChromaDB
            filter_list = [{k: {"$eq": v} if not isinstance(v, dict) else v} for k, v in metadata_filter.items()]
            
            # ChromaDB yêu cầu một toán tử logic duy nhất ở cấp cao nhất nếu có nhiều điều kiện
            if len(filter_list) > 1:
                chroma_where = {"$and": filter_list}
            else:
                chroma_where = filter_list[0]
            
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=chroma_where,
                include=["documents", "metadatas", "embeddings"]
            )
            
            formatted_results = []
            if results and results.get("ids") and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    doc_emb = results["embeddings"][0][i] if results.get("embeddings") else None
                    
                    if doc_emb is not None:
                        if hasattr(doc_emb, "tolist"):
                            score = compute_similarity(query_embedding, doc_emb.tolist())
                        else:
                            score = compute_similarity(query_embedding, doc_emb)
                    else:
                        score = 1.0

                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] or {},
                        "score": float(score)
                    })
            return formatted_results
        else:
            # Lọc trên RAM
            filtered = [
                r for r in self._store 
                if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
            return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        if self._use_chroma and self._collection is not None:
            try:
                count_before = self._collection.count()
                # Thử xóa theo ID trực tiếp (dành cho bộ test id == doc_id)
                self._collection.delete(ids=[doc_id])
                if self._collection.count() < count_before: return True
                
                # Thử xóa theo metadata doc_id
                self._collection.delete(where={"doc_id": doc_id})
                return self._collection.count() < count_before
            except:
                return False
        else:
            initial_len = len(self._store)
            # Lọc bỏ các bản ghi trùng ID HOẶC có metadata doc_id trùng khớp
            self._store = [r for r in self._store if r["id"] != doc_id and r["metadata"].get("doc_id") != doc_id]
            return len(self._store) < initial_len