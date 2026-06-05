from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # Lưu trữ tham chiếu đến vector store và hàm gọi LLM
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Thực hiện quy trình RAG để trả lời câu hỏi.
        """
        # 1. Truy vấn các đoạn văn bản (chunks) liên quan nhất từ Store
        results = self.store.search(query=question, top_k=top_k)
        
        # Nếu không tìm thấy dữ liệu liên quan, ta có thể xử lý riêng hoặc để LLM tự quyết định
        if not results:
            context = "Không có dữ liệu liên quan trong cơ sở tri thức."
        else:
            # Gộp nội dung các chunk tìm được thành một đoạn văn cảnh (context)
            context = "\n---\n".join([res["content"] for res in results])

        # 2. Xây dựng Prompt mẫu (Prompt Engineering)
        # Việc sử dụng cấu trúc rõ ràng giúp LLM không bị lạc đề (Grounding)
        prompt = (
            "Bạn là một trợ lý hữu ích. Hãy sử dụng thông tin trong phần Ngữ cảnh dưới đây "
            "để trả lời câu hỏi của người dùng. Nếu thông tin không có trong ngữ cảnh, "
            "hãy trả lời là bạn không biết, đừng tự bịa ra thông tin.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n"
            "Trả lời:"
        )

        # 3. Gọi hàm LLM và trả về kết quả cuối cùng
        return self.llm_fn(prompt)