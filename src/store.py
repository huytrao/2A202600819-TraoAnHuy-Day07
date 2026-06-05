from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        metadata = dict(doc.metadata) if doc.metadata is not None else {}
        metadata["doc_id"] = doc.id
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records:
            return []
        query_vector = self._embedding_fn(query)
        scored_records = []
        for rec in records:
            score = _dot(query_vector, rec["embedding"])
            scored_records.append({
                "id": rec["id"],
                "content": rec["content"],
                "metadata": rec["metadata"],
                "score": float(score)
            })
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if self._use_chroma and self._collection is not None:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for doc in docs:
                rec = self._make_record(doc)
                ids.append(rec["id"])
                documents.append(rec["content"])
                embeddings.append(rec["embedding"])
                metadatas.append(rec["metadata"])
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        else:
            for doc in docs:
                rec = self._make_record(doc)
                self._store.append(rec)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self.search_with_filter(query, top_k=top_k, metadata_filter=None)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma and self._collection is not None:
            query_vector = self._embedding_fn(query)
            where = None
            if metadata_filter:
                where = metadata_filter
            results = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where
            )
            output = []
            if results and results["documents"]:
                for i in range(len(results["documents"][0])):
                    distance = results["distances"][0][i] if "distances" in results and results["distances"] else 0.0
                    score = 1.0 - distance
                    output.append({
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "score": score
                    })
            output.sort(key=lambda x: x["score"], reverse=True)
            return output[:top_k]
        else:
            filtered_records = []
            for rec in self._store:
                if metadata_filter:
                    match = True
                    for k, v in metadata_filter.items():
                        if rec["metadata"].get(k) != v:
                            match = False
                            break
                    if not match:
                        continue
                filtered_records.append(rec)
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            try:
                existing = self._collection.get(where={"doc_id": doc_id})
                if existing and existing["ids"]:
                    self._collection.delete(ids=existing["ids"])
                    return True
                return False
            except Exception:
                return False
        else:
            before_len = len(self._store)
            self._store = [rec for rec in self._store if rec["metadata"].get("doc_id") != doc_id]
            after_len = len(self._store)
            return after_len < before_len
