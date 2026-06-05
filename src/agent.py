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
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # 1. Retrieve top-k relevant chunks from the store.
        results = self.store.search(question, top_k=top_k)
        
        # 2. Build a prompt with the chunks as context.
        context_parts = []
        for r in results:
            context_parts.append(r["content"])
        context_str = "\n\n".join(context_parts)
        
        prompt = (
            f"Use the following pieces of context to answer the question at the end.\n"
            f"If you don't know the answer, say that you don't know.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        
        # 3. Call the LLM to generate an answer.
        return self.llm_fn(prompt)
