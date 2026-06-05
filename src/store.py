from __future__ import annotations

from importlib import import_module
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
            chromadb = import_module("chromadb")
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata or {})
        metadata.setdefault("doc_id", doc.id)
        metadata.setdefault("source_id", doc.id)

        record = {
            "id": f"{doc.id}-{self._next_index}",
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }
        self._next_index += 1
        return record

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0 or not records:
            return []

        query_embedding = self._embedding_fn(query)
        results: list[dict[str, Any]] = []

        for record in records:
            score = _dot(query_embedding, record["embedding"])
            results.append(
                {
                    "id": record["id"],
                    "content": record["content"],
                    "metadata": dict(record.get("metadata", {})),
                    "score": float(score),
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store.
        """
        if not docs:
            return

        records = [self._make_record(doc) for doc in docs]

        if self._use_chroma and self._collection is not None:
            self._collection.add(
                ids=[record["id"] for record in records],
                documents=[record["content"] for record in records],
                embeddings=[record["embedding"] for record in records],
                metadatas=[record["metadata"] for record in records],
            )
        else:
            self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if top_k <= 0:
            return []

        if self._use_chroma and self._collection is not None:
            if self.get_collection_size() == 0:
                return []

            raw_results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=min(top_k, self.get_collection_size()),
                include=["documents", "metadatas", "distances"],
            )
            return self._format_chroma_results(raw_results)

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return int(self._collection.count())

        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[dict[str, Any]]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            if self.get_collection_size() == 0 or top_k <= 0:
                return []

            raw_results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=min(top_k, self.get_collection_size()),
                where=metadata_filter,
                include=["documents", "metadatas", "distances"],
            )
            return self._format_chroma_results(raw_results)

        filtered_records = [
            record
            for record in self._store
            if all(
                record.get("metadata", {}).get(key) == value
                for key, value in metadata_filter.items()
            )
        ]

        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            existing = self._collection.get(
                where={"doc_id": doc_id},
                include=[],
            )

            ids = existing.get("ids", []) if isinstance(existing, dict) else []

            if not ids:
                return False

            self._collection.delete(ids=ids)
            return True

        original_size = len(self._store)
        self._store = [
            record
            for record in self._store
            if record.get("metadata", {}).get("doc_id") != doc_id
        ]

        return len(self._store) < original_size

    def _format_chroma_results(self, raw_results: dict[str, Any]) -> list[dict[str, Any]]:
        documents = raw_results.get("documents", [])
        metadatas = raw_results.get("metadatas", [])
        distances = raw_results.get("distances", [])

        if not documents:
            return []

        rows: list[dict[str, Any]] = []
        for doc, metadata, distance in zip(
            documents[0],
            metadatas[0] if metadatas else [],
            distances[0] if distances else [],
        ):
            rows.append(
                {
                    "content": doc,
                    "metadata": dict(metadata or {}),
                    "score": -float(distance) if distance is not None else 0.0,
                }
            )

        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows

