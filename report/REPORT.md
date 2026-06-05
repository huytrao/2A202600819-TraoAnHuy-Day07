# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đức Mạnh
**Nhóm:** B4
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (gần 1.0) nghĩa là hai vector embedding có hướng gần giống nhau trong không gian nhiều chiều, tức là hai đoạn văn bản có ý nghĩa ngữ nghĩa tương tự nhau. Giá trị càng gần 1.0 thì hai văn bản càng liên quan về mặt nội dung.

**Ví dụ HIGH similarity:**
- Sentence A: "Python is a popular programming language used for web development."
- Sentence B: "Python is widely used in software development and web applications."
- Tại sao tương đồng: Cả hai câu đều nói về Python như một ngôn ngữ lập trình dùng cho phát triển web, chia sẻ cùng chủ đề và từ khóa chính.

**Ví dụ LOW similarity:**
- Sentence A: "Python is a popular programming language."
- Sentence B: "The weather in Hanoi is very hot today."
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau (lập trình vs thời tiết), không có từ khóa hay ngữ nghĩa chung.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo góc giữa hai vector, không phụ thuộc vào độ dài (magnitude) của vector. Điều này quan trọng vì các text embeddings có thể có magnitude khác nhau tùy thuộc vào độ dài văn bản, nhưng hướng của vector mới thể hiện ý nghĩa ngữ nghĩa. Euclidean distance bị ảnh hưởng bởi magnitude nên kém chính xác hơn khi so sánh semantic similarity.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Phép tính:* step = chunk_size - overlap = 500 - 50 = 450. Số chunks = ceil((10000 - 500) / 450) + 1 = ceil(9500 / 450) + 1 = ceil(21.11) + 1 = 22 + 1 = 23. Chunk cuối bắt đầu tại start = 22 × 450 = 9900, lấy text[9900:10400] nhưng text chỉ có 10000 ký tự nên chunk cuối có 100 ký tự.
> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Với overlap=100, step = 500 - 100 = 400, số chunks = ceil(9500/400) + 1 = 24 + 1 = **25 chunks** — nhiều hơn 23 chunks khi overlap=50. Overlap nhiều hơn giúp bảo toàn ngữ cảnh tại ranh giới giữa các chunks, tránh mất thông tin khi một câu hoặc ý bị cắt đôi giữa hai chunks liền kề.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** RAG System Design & Internal Knowledge Assistant

**Tại sao nhóm chọn domain này?**
> Domain RAG (Retrieval-Augmented Generation) phù hợp với nội dung lab vì chính hệ thống đang xây dựng là một RAG pipeline. Các tài liệu bao gồm hướng dẫn thiết kế RAG, chunking experiment, vector store notes, và customer support playbook — tất cả đều liên quan trực tiếp đến việc xây dựng trợ lý tri thức nội bộ. Domain này cũng có cấu trúc markdown rõ ràng, phù hợp để thử nghiệm các chunking strategies khác nhau.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | techment_enterprise_rag.md | Techment Blog | 30,969 | source="Techment Blog", category=business_blog, time=2026-06-01, access_level=public |
| 2 | llamaindex_concepts.md | LlamaIndex Docs | 5,727 | source="LlamaIndex Docs", category=documentation, time=2026-01-15, access_level=public |
| 3 | anthropic_contextual_rag.md | Anthropic Blog | 17,364 | source="Anthropic Blog", category=research_blog, time=2026-05-20, access_level=public |
| 4 | rag_comprehensive_guide.md | Internal Guide | 8,231 | source="Internal Guide", category=technical_guide, time=2026-06-05, access_level=internal |
| 5 | vector_store_notes.md | Class Notes | 2,123 | source="Class Notes", category=documentation, time=2026-06-05, access_level=public |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | str | "Anthropic Blog", "LlamaIndex Docs" | Truy vết nguồn gốc chunk, giúp user biết câu trả lời đến từ đâu |
| category | str | "research_blog", "documentation", "technical_guide" | Phân loại loại tài liệu để filter theo use case (blog vs docs vs guide) |
| time | str | "2026-06-05" | Lọc theo thời gian để ưu tiên tài liệu mới nhất |
| access_level | str | "public", "internal" | Kiểm soát quyền truy cập — chỉ trả về tài liệu user được phép xem |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu (chunk_size=200):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| techment_enterprise_rag.md | FixedSizeChunker (`fixed_size`) | 152 | 199.4 | Không — cắt giữa câu |
| techment_enterprise_rag.md | SentenceChunker (`by_sentences`) | 38 | 792.6 | Có nhưng chunk rất dài |
| techment_enterprise_rag.md | RecursiveChunker (`recursive`) | 211 | 141.8 | Tốt — tách theo paragraph |
| anthropic_contextual_rag.md | FixedSizeChunker (`fixed_size`) | 87 | 199.0 | Không — cắt giữa câu |
| anthropic_contextual_rag.md | SentenceChunker (`by_sentences`) | 40 | 430.4 | Có — giữ nguyên câu |
| anthropic_contextual_rag.md | RecursiveChunker (`recursive`) | 137 | 124.6 | Tốt — tách theo paragraph |
| rag_comprehensive_guide.md | FixedSizeChunker (`fixed_size`) | 42 | 196.0 | Không — cắt giữa câu |
| rag_comprehensive_guide.md | SentenceChunker (`by_sentences`) | 25 | 327.4 | Có — giữ nguyên câu |
| rag_comprehensive_guide.md | RecursiveChunker (`recursive`) | 67 | 121.4 | Tốt — tách theo heading |

### Strategy Của Tôi

**Loại:** MarkdownHeaderChunker (custom strategy)

**Mô tả cách hoạt động:**
> MarkdownHeaderChunker tách markdown text theo headers (# ## ### ####), giữ nguyên ranh giới section. Mỗi chunk chứa một section hoàn chỉnh với header. Nếu section quá lớn (> chunk_size), fall back sang RecursiveChunker bên trong section đó. Các section quá nhỏ (< min_chunk_size) được merge với section tiếp theo. Đặc biệt, parent headers được prepend vào mỗi chunk để giữ context hierarchy (ví dụ: "# Main > ## Sub" trước nội dung sub-section).

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu RAG/technical documentation có cấu trúc markdown rõ ràng với heading hierarchy. MarkdownHeaderChunker khai thác cấu trúc này trực tiếp — mỗi chunk tương ứng với một section có ý nghĩa, và parent header context giúp embedding model hiểu chunk thuộc topic nào. Benchmark cho thấy MarkdownHeaderChunker(500) đạt hit rate 40% (2/5), cao nhất trong tất cả strategies.

**Code snippet (nếu custom):**
```python
class MarkdownHeaderChunker:
    def __init__(self, chunk_size=500, min_chunk_size=50):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(self, text):
        if not text:
            return []
        sections = re.split(r'(?=^#{1,4}\s)', text, flags=re.MULTILINE)
        sections = [s.strip() for s in sections if s.strip()]
        chunks, header_stack = [], []
        for section in sections:
            header_match = re.match(r'^(#{1,4})\s+(.+)', section)
            if header_match:
                level = len(header_match.group(1))
                header_stack = [(l, h) for l, h in header_stack if l < level]
                header_stack.append((level, header_match.group(0).strip()))
            context_prefix = " > ".join(h for _, h in header_stack[:-1]) + "\n" if header_match and len(header_stack) > 1 else ""
            full_chunk = context_prefix + section
            if len(full_chunk) <= self.chunk_size:
                chunks.append(full_chunk)
            else:
                for sc in RecursiveChunker(chunk_size=self.chunk_size).chunk(section):
                    chunks.append((context_prefix + sc) if context_prefix else sc)
        # Merge small chunks
        merged = []
        for c in chunks:
            if merged and len(merged[-1]) < self.min_chunk_size:
                merged[-1] = merged[-1] + "\n\n" + c
            else:
                merged.append(c)
        return merged
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| All docs (benchmark) | best baseline (RecursiveChunker 500) | 232 chunks | ~140 | Hit rate 20% (1/5) |
| All docs (benchmark) | **MarkdownHeaderChunker (500)** | **256 chunks** | ~250 | **Hit rate 40% (2/5)** |

### So Sánh Với Thành Viên Khác

**Strategy nào tốt nhất cho domain này? Tại sao?**
> MarkdownHeaderChunker là lựa chọn tốt nhất cho domain technical documentation/RAG vì tài liệu có cấu trúc markdown rõ ràng. Strategy này tận dụng heading hierarchy để tạo chunk có ý nghĩa theo section, trong khi FixedSizeChunker cắt cứng và SentenceChunker tạo chunk quá dài. Benchmark xác nhận: hit rate 40% vs 20% của các baseline.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng regex `(?<=\.)\s+|(?<=!)\s+|(?<=\?)\s+` để split text tại các vị trí sau dấu `.`, `!`, `?` theo sau bởi whitespace (lookbehind). Sau khi split, các câu được strip whitespace và lọc bỏ phần tử rỗng, rồi nhóm lại theo `max_sentences_per_chunk`, mỗi nhóm join bằng dấu cách. Edge case: text rỗng trả về list rỗng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm đệ quy: thử split text bằng separator đầu tiên trong danh sách ưu tiên (`\n\n` > `\n` > `. ` > ` ` > `""`), merge các phần nhỏ lại với nhau cho đến khi vượt `chunk_size`, rồi đệ quy split tiếp các chunk còn quá lớn bằng separator tiếp theo. Base case: text ≤ chunk_size trả về nguyên, hết separator thì force-split theo chunk_size.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document được embed bằng `embedding_fn` và lưu dưới dạng dict chứa `id`, `content`, `embedding`, `metadata` vào list `_store` (in-memory) hoặc ChromaDB collection. Khi search, query được embed rồi tính dot product với tất cả stored embeddings, sort giảm dần theo score và trả về top-k kết quả.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` filter **trước** khi search: lọc các record trong `_store` theo metadata_filter (tất cả key-value phải match), rồi chạy similarity search trên tập đã lọc. `delete_document` xóa tất cả record có `metadata['doc_id']` bằng `doc_id` được chỉ định, trả về `True` nếu có record bị xóa, `False` nếu không tìm thấy.

### KnowledgeBaseAgent

**`answer`** — approach:
> Retrieve top-k chunks liên quan nhất từ store bằng `store.search()`. Build prompt theo pattern RAG: inject các chunks vào phần "Context", kèm câu hỏi vào phần "Question", rồi gọi `llm_fn(prompt)` để sinh câu trả lời. Prompt structure: `"Use the following context to answer the question.\n\nContext:\n{chunks}\n\nQuestion: {question}\n\nAnswer:"`.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

42 passed in 0.19s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)


| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | What are the five categories of data-backed LLM applications? | LlamaIndex defines five types of LLM apps using data | high | 0.1873 | Không — thấp hơn dự đoán |
| 2 | Contextual Embeddings reduce retrieval failure rate | Combining contextual BM25 improves top-20 chunk retrieval | high | -0.0825 | Không — âm dù cùng chủ đề |
| 3 | The weather is sunny today | Machine learning model training process | low | -0.1903 | Đúng — âm, rất khác |
| 4 | Euclidean Distance drawback in vector databases | Cosine Similarity is better than Euclidean for text search | high | 0.1015 | Không — thấp dù liên quan |
| 5 | Triad of Metrics in Ragas framework | Faithfulness, Answer Relevance, and Context Relevancy | high | -0.2068 | Không — âm dù liên quan |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 4 và 5 bất ngờ nhất: hai câu có cùng chủ đề (vector distance metrics, Ragas metrics) nhưng score gần 0 hoặc âm. Điều này cho thấy MockEmbedder (hash-based) không hiểu ngữ nghĩa thực sự — nó chỉ tạo vector dựa trên chuỗi ký tự, không phải ý nghĩa. Với embedding model thực (như all-MiniLM-L6-v2), các cặp câu đồng nghĩa sẽ có score cao hơn nhiều. Đây là lý do benchmark hit rate thấp khi dùng MockEmbedder.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (từ benchmark.json — nhóm thống nhất)

| # | Query | Gold Answer | Expected Source |
|---|-------|-------------|-----------------|
| 1 | According to LlamaIndex, what are the five categories of data-backed LLM applications? | Agents, Workflows, Structured Data Extraction, Query Engines, and Chat Engines | llamaindex_concepts.md |
| 2 | How much does combining Contextual Embeddings and Contextual BM25 reduce the top-20-chunk retrieval failure rate? | It reduces the top-20-chunk retrieval failure rate by 49% | anthropic_contextual_rag.md |
| 3 | In a Vector Database, what is the main drawback of using Euclidean Distance compared to Cosine Similarity? | If a text chunk is very long and contains repeated words, the vector length increases, pushing the endpoint away and skewing results | rag_comprehensive_guide.md |
| 4 | Based on the internal architecture guide, what are the three specific metrics used to measure the 'Search' phase (Retrieval KPIs)? | Hit Rate, Mean Reciprocal Rank (MRR), and NDCG | rag_definitive_architecture.md |
| 5 | What are the three components of the 'Triad of Metrics' in the Ragas framework? | Faithfulness, Answer Relevance, and Context Relevancy | rag_definitive_architecture.md |

### Kết Quả Của Tôi
| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Hit? |
|---|-------|--------------------------------|-------|-----------|------|
| 1 | Five categories of LLM apps (LlamaIndex) | "# Comprehensive Technical Guide: RAG Architecture, Chunking Techniques, and Vect..." (rag_comprehensive_guide.md) | 0.2855 | Không — top-1 sai source, expected llamaindex_concepts.md | MISS |
| 2 | Contextual Embeddings + BM25 failure rate reduction | "## RAG in 2026 for Enterprises: What Business and Technology Leaders Must Know..." (techment_enterprise_rag.md) | 0.3300 | Không — top-1 sai source, nhưng anthropic_contextual_rag.md xuất hiện ở top-5 | Hit@5 |
| 3 | Euclidean Distance drawback vs Cosine Similarity | "New advances (hybrid retrieval, multimodal RAG, rerankers) make deployment e..." (techment_enterprise_rag.md) | 0.3590 | Không — top-1 sai source, nhưng rag_comprehensive_guide.md xuất hiện ở top-5 | Hit@5 |
| 4 | Three Retrieval KPIs (internal architecture guide) | Không có kết quả — metadata filter `{"category": "architecture", "access": "internal"}` không match | 0.0000 | Không — metadata filter fail, store trống sau filter | MISS |
| 5 | Triad of Metrics in Ragas framework | "Search across Confluence, SharePoint, Jira — Knowledge bots for engineer..." (techment_enterprise_rag.md) | 0.3781 | Không — top-1 sai source, expected rag_definitive_architecture.md | MISS |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 0 / 5 trong top-3 (2/5 trong top-5: q2 Hit@5, q3 Hit@5)

**Phân tích:**
- **Hit rate:** 2/5 = 40% (top-5), 0/5 = 0% (top-3). MarkdownHeaderChunker(500) vẫn là strategy tốt nhất (40% vs 20% của tất cả baseline).
- **Q4 fail hoàn toàn:** metadata filter `{"category": "architecture", "access": "internal"}` không match với metadata schema trong `metadata.json` — cần align metadata keys giữa benchmark và data.
- **Root cause:** MockEmbedder (hash-based) không hiểu ngữ nghĩa, nên retrieval gần như random. Với real embedding model (sentence-transformers), kết quả sẽ cải thiện đáng kể vì model hiểu semantic similarity giữa query và chunk content.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Học được rằng việc so sánh chunking strategies trên cùng một dataset cho thấy rõ trade-off giữa chunk size và context preservation. Thành viên dùng SentenceChunker cho thấy chunk giữ nguyên câu dễ đọc hơn khi inspect thủ công, nhưng chunk size không đều gây khó khăn cho embedding model.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhận ra rằng MockEmbedder chỉ phù hợp cho testing — kết quả retrieval với mock embeddings không phản ánh chất lượng thực tế. Các nhóm dùng real embedding model (sentence-transformers) có retrieval quality tốt hơn đáng kể, đặc biệt với các câu hỏi paraphrase.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Sẽ gán metadata chi tiết hơn (doc_type, language, section_title) ngay từ đầu để tận dụng search_with_filter. Ngoài ra sẽ thử dùng real embedding model thay vì MockEmbedder để đánh giá retrieval quality chính xác hơn, và thêm overlap vào RecursiveChunker để tránh mất context tại ranh giới chunk.

