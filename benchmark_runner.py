"""
benchmark_runner.py — Day 7 Lab Benchmark Script
=================================================
Usage:
    python benchmark_runner.py

What it does:
1.  Reads .env to pick EMBEDDING_PROVIDER (mock | local | openai)
2.  Reads data/metadata.json   → maps filename → metadata dict
3.  Loads each .md / .txt file from data/
4.  Chunks every document with SemanticHeaderChunker (our new strategy)
5.  Adds chunks to EmbeddingStore with the selected embedder
6.  Runs 5 benchmark queries from benchmark/benchmark.json
7.  Prints per-query results (top-3 chunks, relevance check, score)
8.  Runs ChunkingStrategyComparator on techment_enterprise_rag.md
    comparing all 4 strategies side-by-side
9.  Prints everything in a format ready to paste into REPORT.md
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

# Force UTF-8 output on Windows so Unicode chars don't crash cp1258
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Resolve project root so we can import `src` without installing the package.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Load .env manually (no python-dotenv required)
# ---------------------------------------------------------------------------
_ENV_FILE = ROOT / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#"):
            continue
        if "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

from src import (
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    SemanticHeaderChunker,
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
)


def _build_embedder():
    """Select embedder based on EMBEDDING_PROVIDER env var (reads from .env)."""
    provider = os.environ.get(EMBEDDING_PROVIDER_ENV, "mock").lower().strip()
    if provider == "local":
        print(f"[Embedder] Using LocalEmbedder ({LOCAL_EMBEDDING_MODEL}) — loading model...")
        return LocalEmbedder()
    elif provider == "openai":
        print("[Embedder] Using OpenAIEmbedder")
        return OpenAIEmbedder()
    else:
        print("[Embedder] Using MockEmbedder (hash-based, not semantic)")
        return MockEmbedder(dim=64)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
BENCHMARK_FILE = ROOT / "benchmark" / "benchmark.json"
METADATA_FILE = DATA_DIR / "metadata.json"
CHUNK_SIZE = 500
OVERLAP = 0

# Map benchmark expected_source names (with file2_, file3_…) to real filenames
SOURCE_MAP = {
    "file2_llamaindex_concepts.md": "llamaindex_concepts.md",
    "file3_anthropic_contextual_rag.md": "anthropic_contextual_rag.md",
    "file5_rag_comprehensive_guide.md": "rag_comprehensive_guide.md",
    "file6_rag_definitive_architecture.md": "rag_definitive_architecture.md",
    "file1_techment_enterprise_rag.md": "techment_enterprise_rag.md",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(title: str = "", width: int = 70) -> str:
    if title:
        pad = (width - len(title) - 2) // 2
        return "\n" + "-" * pad + f" {title} " + "-" * pad
    return "\n" + "-" * width


def _wrap(text: str, indent: int = 4, width: int = 90) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=prefix)


def load_metadata() -> dict[str, dict]:
    """Return {filename: metadata_dict}."""
    with open(METADATA_FILE, encoding="utf-8") as fh:
        entries = json.load(fh)
    return {e["file_name"]: e["metadata"] for e in entries}


def load_documents(metadata_map: dict[str, dict]) -> list[Document]:
    """Load all files listed in metadata.json, return as Document objects."""
    docs: list[Document] = []
    for fname, meta in metadata_map.items():
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"  [WARN] File not found, skipping: {fpath}")
            continue
        text = fpath.read_text(encoding="utf-8")
        doc_id = fpath.stem  # e.g. "llamaindex_concepts"
        meta_copy = dict(meta)
        meta_copy["doc_id"] = doc_id
        meta_copy["file_name"] = fname
        docs.append(Document(id=doc_id, content=text, metadata=meta_copy))
    return docs


def chunk_documents(docs: list[Document], chunker: SemanticHeaderChunker) -> list[Document]:
    """Split each document into chunk-level Document objects."""
    chunks: list[Document] = []
    for doc in docs:
        pieces = chunker.chunk(doc.content)
        for i, piece in enumerate(pieces):
            chunk_meta = dict(doc.metadata)
            chunk_meta["chunk_index"] = i
            chunk_meta["chunk_total"] = len(pieces)
            chunks.append(
                Document(
                    id=f"{doc.id}_chunk_{i}",
                    content=piece,
                    metadata=chunk_meta,
                )
            )
    return chunks


def check_relevance(chunk_content: str, gold_answer: str) -> bool:
    """
    Simple lexical relevance check:
    Count how many distinct key words from the gold answer appear in the chunk.
    If >= 3 key words match → consider relevant.
    """
    gold_words = set(gold_answer.lower().split())
    # Keep only words longer than 4 chars as "key" words
    key_words = {w.strip(".,;:()[]\"'") for w in gold_words if len(w) > 4}
    chunk_lower = chunk_content.lower()
    matches = sum(1 for w in key_words if w in chunk_lower)
    return matches >= 3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    provider = os.environ.get(EMBEDDING_PROVIDER_ENV, "mock").lower()
    print(_sep("DAY 7 BENCHMARK RUNNER", width=70))
    print(f"Strategy  : SemanticHeaderChunker")
    print(f"Chunk size: {CHUNK_SIZE}  |  Overlap: {OVERLAP}")
    print(f"Embedder  : {provider}")

    # ------------------------------------------------------------------
    # 1. Load documents
    # ------------------------------------------------------------------
    print(_sep("Loading Documents"))
    metadata_map = load_metadata()
    docs = load_documents(metadata_map)

    print(f"\n{'#':<3} {'File':<40} {'Chars':>7} {'Category'}")
    print("─" * 70)
    for i, doc in enumerate(docs, 1):
        print(
            f"{i:<3} {doc.metadata['file_name']:<40} "
            f"{len(doc.content):>7,}  {doc.metadata.get('category', 'n/a')}"
        )

    # ------------------------------------------------------------------
    # 2. Chunk with SemanticHeaderChunker
    # ------------------------------------------------------------------
    print(_sep("Chunking with SemanticHeaderChunker"))
    chunker = SemanticHeaderChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    chunk_docs = chunk_documents(docs, chunker)

    print(f"\n{'#':<3} {'Doc ID':<35} {'Chunks':>6}")
    print("─" * 50)
    doc_chunk_counts: dict[str, int] = {}
    for chunk in chunk_docs:
        base = chunk.metadata["doc_id"]
        doc_chunk_counts[base] = doc_chunk_counts.get(base, 0) + 1
    for i, (doc_id, count) in enumerate(doc_chunk_counts.items(), 1):
        print(f"{i:<3} {doc_id:<35} {count:>6}")
    print(f"\nTotal chunks stored: {len(chunk_docs)}")

    # ------------------------------------------------------------------
    # 3. Build EmbeddingStore
    # ------------------------------------------------------------------
    print(_sep("Building EmbeddingStore"))
    embedder = _build_embedder()
    store = EmbeddingStore(collection_name="benchmark_run", embedding_fn=embedder)
    print("Embedding all chunks... (this may take a moment with LocalEmbedder)")
    store.add_documents(chunk_docs)
    print(f"Store size: {store.get_collection_size()} chunks")

    # ------------------------------------------------------------------
    # 4. Run benchmark queries
    # ------------------------------------------------------------------
    print(_sep("Benchmark Queries"))
    with open(BENCHMARK_FILE, encoding="utf-8") as fh:
        queries = json.load(fh)

    relevant_in_top3 = 0
    query_results = []

    for q in queries:
        qid = q["id"]
        query_text = q["query"]
        gold = q["gold_answer"]
        expected_src_raw = q.get("expected_source", "")
        expected_src = SOURCE_MAP.get(expected_src_raw, expected_src_raw)
        meta_filter = q.get("metadata_filter", {}) or None

        # Run search
        if meta_filter:
            results = store.search_with_filter(query_text, top_k=3, metadata_filter=meta_filter)
        else:
            results = store.search(query_text, top_k=3)

        # Check top-1
        top1 = results[0] if results else None
        top1_content = top1["content"] if top1 else ""
        top1_score = top1["score"] if top1 else 0.0
        top1_source = top1["metadata"].get("file_name", "?") if top1 else "?"

        # Check relevance for top-3
        any_relevant = any(check_relevance(r["content"], gold) for r in results)
        if any_relevant:
            relevant_in_top3 += 1

        top1_relevant = check_relevance(top1_content, gold)

        query_results.append({
            "id": qid,
            "query": query_text,
            "gold": gold,
            "top1_content": top1_content,
            "top1_score": top1_score,
            "top1_source": top1_source,
            "top1_relevant": top1_relevant,
            "any_top3_relevant": any_relevant,
            "results": results,
        })

        # Print per-query report
        print(f"\n{'═' * 68}")
        print(f"[{qid.upper()}] {query_text}")
        print(f"{'=' * 68}")
        print(f"Gold answer: {gold}")
        print(f"Expected src: {expected_src}")
        print()
        for rank, res in enumerate(results, 1):
            rel_flag = "[OK] relevant" if check_relevance(res["content"], gold) else "[--] not relevant"
            print(f"  Top-{rank} | score={res['score']:.4f} | {rel_flag}")
            print(f"          src: {res['metadata'].get('file_name', '?')}")
            snippet = res["content"][:200].replace("\n", " ")
            print(f"          snippet: {snippet}...")
        print()
        print(f"  -> Top-3 any relevant: {'YES [OK]' if any_relevant else 'NO [--]'}")

    print(_sep("Benchmark Summary"))
    print(f"\nQueries with relevant chunk in top-3: {relevant_in_top3} / {len(queries)}")

    # ------------------------------------------------------------------
    # 5. Comparator analysis on techment_enterprise_rag.md
    # ------------------------------------------------------------------
    print(_sep("ChunkingStrategyComparator — techment_enterprise_rag.md"))
    techment_path = DATA_DIR / "techment_enterprise_rag.md"
    if techment_path.exists():
        techment_text = techment_path.read_text(encoding="utf-8")
        comparator = ChunkingStrategyComparator(chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        comp_results = comparator.compare(techment_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

        print(f"\n{'Strategy':<20} {'Chunks':>7} {'Avg Len':>9} {'Min Len':>8} {'Max Len':>8}")
        print("─" * 60)
        for strategy, stats in comp_results.items():
            print(
                f"{strategy:<20} {stats['count']:>7} "
                f"{stats['avg_length']:>9.1f} "
                f"{stats['min_length']:>8} "
                f"{stats['max_length']:>8}"
            )
    else:
        print("  [WARN] techment_enterprise_rag.md not found, skipping comparator.")
        comp_results = {}

    # ------------------------------------------------------------------
    # 6. Print REPORT-ready tables
    # ------------------------------------------------------------------
    print(_sep("REPORT.md — Ready-to-Paste Data"))

    print("\n### §2 Data Inventory (from metadata.json)\n")
    print("| # | File | Source | Chars | Category |")
    print("|---|------|--------|-------|----------|")
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        print(
            f"| {i} | {meta['file_name']} | {meta['source']} "
            f"| {len(doc.content):,} | {meta['category']} |"
        )

    print("\n### §6 Benchmark Queries & Gold Answers\n")
    print("| # | Query | Gold Answer |")
    print("|---|-------|-------------|")
    for i, q in enumerate(queries, 1):
        g = q["gold_answer"].replace("|", "\\|")
        print(f"| {i} | {q['query']} | {g} |")

    print("\n### §6 My Results (SemanticHeaderChunker)\n")
    print("| # | Query | Top-1 Chunk Snippet | Score | Relevant? |")
    print("|---|-------|---------------------|-------|-----------|")
    for i, r in enumerate(query_results, 1):
        snippet = r["top1_content"][:80].replace("\n", " ").replace("|", "\\|") + "…"
        rel = "Yes" if r["top1_relevant"] else "No"
        print(f"| {i} | {r['query'][:60]}… | {snippet} | {r['top1_score']:.4f} | {rel} |")
    print(f"\n**Relevant in top-3: {relevant_in_top3} / {len(queries)}**")

    if comp_results:
        print("\n### §3 Baseline Analysis (techment_enterprise_rag.md, chunk_size=500)\n")
        print("| Strategy | Chunks | Avg Length | Min | Max | Context Preservation |")
        print("|----------|--------|------------|-----|-----|----------------------|")
        notes = {
            "fixed_size": "Poor — cuts mid-sentence",
            "by_sentences": "Good — preserves sentences",
            "recursive": "Very Good — respects structure",
            "semantic_header": "Excellent — keeps sections intact",
        }
        for strategy, stats in comp_results.items():
            note = notes.get(strategy, "")
            print(
                f"| {strategy} | {stats['count']} "
                f"| {stats['avg_length']:.1f} "
                f"| {stats['min_length']} "
                f"| {stats['max_length']} "
                f"| {note} |"
            )

    print(_sep("DONE", width=70))


if __name__ == "__main__":
    main()
