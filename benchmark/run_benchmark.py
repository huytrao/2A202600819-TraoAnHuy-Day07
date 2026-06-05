"""Benchmark all chunking strategies against benchmark.json queries."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import FixedSizeChunker, SentenceChunker, RecursiveChunker, MarkdownHeaderChunker
from src.embeddings import _mock_embed, LocalEmbedder
from src.models import Document
from src.store import EmbeddingStore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
METADATA_FILE = DATA_DIR / "metadata.json"
BENCHMARK_FILE = Path(__file__).resolve().parent / "benchmark.json"

# Map expected_source in benchmark to actual filenames
SOURCE_MAP = {
    "file1_techment_enterprise_rag.md": "techment_enterprise_rag.md",
    "file2_llamaindex_concepts.md": "llamaindex_concepts.md",
    "file3_anthropic_contextual_rag.md": "anthropic_contextual_rag.md",
    "file4_vector_store_notes.md": "vector_store_notes.md",
    "file5_rag_comprehensive_guide.md": "rag_comprehensive_guide.md",
    "file6_rag_definitive_architecture.md": "rag_definitive_architecture.md",
}


def load_all_data_files() -> dict[str, str]:
    """Load all .md and .txt files from data/."""
    texts = {}
    for f in DATA_DIR.iterdir():
        if f.suffix in (".md", ".txt") and f.name != ".gitkeep":
            texts[f.name] = f.read_text(encoding="utf-8")
    return texts


def load_metadata() -> dict[str, dict]:
    """Load metadata from metadata.json, keyed by file_name."""
    if not METADATA_FILE.exists():
        return {}
    with open(METADATA_FILE) as f:
        entries = json.load(f)
    return {e["file_name"]: e.get("metadata", {}) for e in entries}


def load_benchmark() -> list[dict]:
    with open(BENCHMARK_FILE) as f:
        return json.load(f)


def build_store(chunker, texts: dict[str, str], metadata_map: dict[str, dict], embedder) -> EmbeddingStore:
    """Chunk all documents and add to a fresh store."""
    store = EmbeddingStore(collection_name="bench", embedding_fn=embedder)
    docs = []
    for fname, content in texts.items():
        chunks = chunker.chunk(content)
        meta = metadata_map.get(fname, {})
        for i, chunk_text in enumerate(chunks):
            docs.append(Document(
                id=f"{fname}_chunk{i}",
                content=chunk_text,
                metadata={"source_file": fname, "chunk_index": i, **meta},
            ))
    store.add_documents(docs)
    return store


def evaluate_query(store: EmbeddingStore, query: dict, top_k: int = 5) -> dict:
    """Evaluate a single benchmark query. Returns hit info."""
    q_text = query["query"]
    expected_src = query["expected_source"]
    actual_fname = SOURCE_MAP.get(expected_src, expected_src)
    meta_filter = query.get("metadata_filter", {})

    if meta_filter:
        results = store.search_with_filter(q_text, top_k=top_k, metadata_filter=meta_filter)
    else:
        results = store.search(q_text, top_k=top_k)

    # Check if expected source appears in top-k
    hit_at = None
    for i, r in enumerate(results):
        if r["metadata"].get("source_file") == actual_fname:
            hit_at = i + 1
            break

    return {
        "id": query["id"],
        "query": q_text[:80],
        "expected": actual_fname,
        "gold_answer": query["gold_answer"][:80],
        "hit_at": hit_at,
        "top1_source": results[0]["metadata"].get("source_file", "?") if results else "?",
        "top1_score": results[0]["score"] if results else 0,
        "top1_preview": results[0]["content"][:100].replace("\n", " ") if results else "",
    }


def run_benchmark_for_chunker(name: str, chunker, texts, metadata_map, embedder, benchmark, top_k=5):
    """Run full benchmark for one chunker strategy."""
    store = build_store(chunker, texts, metadata_map, embedder)
    print(f"\n{'='*70}")
    print(f"Strategy: {name} | Store size: {store.get_collection_size()} chunks")
    print(f"{'='*70}")

    hits = 0
    results = []
    for q in benchmark:
        r = evaluate_query(store, q, top_k=top_k)
        results.append(r)
        hit_str = f"Hit@{r['hit_at']}" if r['hit_at'] else "MISS"
        if r['hit_at']:
            hits += 1
        print(f"  {r['id']}: {hit_str:8s} | score={r['top1_score']:.4f} | top1={r['top1_source']}")
        print(f"           Q: {r['query']}")
        print(f"           Top1: {r['top1_preview'][:80]}...")

    hit_rate = hits / len(benchmark) * 100
    print(f"\n  Hit Rate: {hits}/{len(benchmark)} = {hit_rate:.1f}%")
    return {"name": name, "hits": hits, "total": len(benchmark), "hit_rate": hit_rate, "results": results}


def main():
    texts = load_all_data_files()
    metadata_map = load_metadata()
    benchmark = load_benchmark()

    print(f"Loaded {len(texts)} files, {len(benchmark)} benchmark queries")

    # Try to use local embedder, fall back to mock
    try:
        embedder = LocalEmbedder()
        print(f"Using embedder: {embedder._backend_name}")
    except Exception:
        embedder = _mock_embed
        print("Using embedder: MockEmbedder (fallback)")

    chunkers = {
        "FixedSizeChunker(500)": FixedSizeChunker(chunk_size=500, overlap=50),
        "SentenceChunker(3)": SentenceChunker(max_sentences_per_chunk=3),
        "RecursiveChunker(500)": RecursiveChunker(chunk_size=500),
        "MarkdownHeaderChunker(500)": MarkdownHeaderChunker(chunk_size=500, min_chunk_size=50),
        "MarkdownHeaderChunker(800)": MarkdownHeaderChunker(chunk_size=800, min_chunk_size=100),
    }

    all_results = []
    for name, chunker in chunkers.items():
        r = run_benchmark_for_chunker(name, chunker, texts, metadata_map, embedder, benchmark)
        all_results.append(r)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Strategy':<30s} {'Hits':>5s} {'Total':>5s} {'Hit Rate':>10s}")
    print("-" * 55)
    for r in all_results:
        print(f"{r['name']:<30s} {r['hits']:>5d} {r['total']:>5d} {r['hit_rate']:>9.1f}%")


if __name__ == "__main__":
    main()
