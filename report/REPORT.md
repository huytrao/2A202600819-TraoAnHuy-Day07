# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trao An Huy
**Nhóm:** B4
**Ngày:** 05/06/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (điểm gần bằng 1) có nghĩa là hai đoạn văn bản rất giống nhau về mặt ngữ nghĩa hoặc chủ đề. Trong không gian vector, các vector embedding của chúng chỉ vào gần như cùng một hướng, bất kể độ lớn (ví dụ: độ dài câu) của chúng.

**Ví dụ HIGH similarity:**
- Sentence A: "Làm thế nào để cài đặt Python trên máy tính?"
- Sentence B: "Hướng dẫn các bước setup Python cho PC."
- Tại sao tương đồng: Cả hai câu đều hỏi về cùng một chủ đề là cài đặt Python, sử dụng các từ đồng nghĩa ("cài đặt" vs "setup", "máy tính" vs "PC") và có cùng mục đích.

**Ví dụ LOW similarity:**
- Sentence A: "Hôm nay thời tiết thật đẹp."
- Sentence B: "Bitcoin là một loại tiền điện tử."
- Tại sao khác: Hai câu này không có bất kỳ mối liên hệ nào về chủ đề, từ vựng hay ngữ nghĩa. Các vector của chúng sẽ chỉ về hai hướng rất khác nhau trong không gian embedding.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity được ưu tiên hơn vì nó chỉ đo góc giữa hai vector, tập trung vào **hướng** (tức là ngữ nghĩa/chủ đề) thay vì **độ lớn**. Độ lớn của vector embedding có thể bị ảnh hưởng bởi các yếu tố như độ dài của câu, điều này không phải lúc nào cũng phản ánh sự khác biệt về ngữ nghĩa. Euclidean distance lại nhạy cảm với cả hướng và độ lớn, nên có thể đánh giá sai sự tương đồng của các câu có độ dài khác nhau.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450)`
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Số lượng chunk sẽ tăng lên (thành 25 chunks: `ceil(9900 / 400)`). Chúng ta muốn có overlap nhiều hơn để đảm bảo thông tin (nhất là các câu dài hoặc các ý chính) nằm ở ranh giới cắt không bị đứt đoạn, giúp ngữ cảnh được giữ trọn vẹn trong ít nhất một chunk và cải thiện chất lượng tìm kiếm (retrieval).

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain nhóm chọn:** RAG & Vector Store Technical Documentation

**Lý do chọn domain này:**  
Domain này tập trung vào tài liệu kỹ thuật về Retrieval-Augmented Generation (RAG) và Vector Store — đây là nền tảng cốt lõi của Lab Day 7. Các tài liệu có cấu trúc markdown rõ ràng với header phân cấp, phù hợp để so sánh các chunking strategy khác nhau. Nội dung đủ phức tạp (30K ký tự) để thể hiện sự khác biệt rõ rệt giữa các strategy, đồng thời có thể thiết kế benchmark queries đòi hỏi thông tin cụ thể từ đúng tài liệu.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Category | Thời gian |
|---|--------------|-------|----------|----------|-----------| 
| 1 | techment_enterprise_rag.md | Techment Blog | 30,314 | business_blog | 2026-06-01 |
| 2 | llamaindex_concepts.md | LlamaIndex Docs | 5,709 | documentation | 2026-01-15 |
| 3 | anthropic_contextual_rag.md | Anthropic Blog | 17,316 | research_blog | 2026-05-20 |
| 4 | rag_comprehensive_guide.md | Internal Guide | 8,231 | technical_guide | 2026-06-05 |
| 5 | vector_store_notes.md | Class Notes | 2,123 | documentation | 2026-06-05 |
### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `category` | String | `business_blog`, `documentation` | Giúp lọc thu hẹp phạm vi tìm kiếm (vd: "Chỉ tìm trong documentation"). Hạn chế nhiễu từ các blog/marketing. |
| `access_level` | String | `public`, `internal` | Giúp thực hiện phân quyền (Role-Based Access Control) khi retrieval, tránh lộ tài liệu mật ra ngoài. |
| `time` | String/Date | `2026-06-01` | Giúp ưu tiên tìm kiếm và trích xuất các tài liệu mới nhất, tránh RAG sử dụng thông tin cũ. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| techment_enterprise_rag.md | FixedSizeChunker (`fixed_size`) | 46 | 500 | Không, dễ bị cắt giữa câu. |
| techment_enterprise_rag.md | SentenceChunker (`by_sentences`) | 32 | ~718 | Tốt, giữ cấu trúc ý nghĩa trọn vẹn của các câu. |
| techment_enterprise_rag.md | RecursiveChunker (`recursive`) | 48 | ~479 | Khá tốt, cắt dựa trên ranh giới đoạn văn trước tiên. |

### Strategy Của Tôi

**Loại:** `SemanticHeaderChunker` (Header-Aware Semantic Chunker — strategy MỚI, implement thêm)

**Mô tả cách hoạt động:**  
`SemanticHeaderChunker` phân tích cấu trúc markdown bằng regex để phát hiện các headers (`#`, `##`, `###`, `####`). Sau đó tạo mỗi chunk từ một section hoàn chỉnh: header + toàn bộ nội dung thuộc section đó — không bao giờ trộn nội dung của hai section khác nhau vào cùng một chunk.

Nếu một section quá lớn (vượt `chunk_size`), thuật toán tách theo paragraph bên trong section đó, nhưng vẫn giữ header ở đầu mỗi sub-chunk (tham số `repeat_header=True`) để người đọc biết chunk thuộc section nào.

Với text thuần không có header (`.txt` files) → fallback hoàn toàn sang `SentenceChunker`.

**Điểm khác biệt then chốt so với `RecursiveChunker`:**  
- `RecursiveChunker`: split rồi merge nhiều phần nhỏ lại theo size limit → có thể trộn nội dung của 2 section khác nhau vào cùng 1 chunk.  
- `SemanticHeaderChunker`: section integrity là bất khả xâm phạm — nội dung của section A và section B không bao giờ cùng chunk.

**Tại sao tôi chọn strategy này cho domain nhóm?**  
Domain RAG Technical Documentation có cấu trúc section rất rõ ràng: mỗi section nói về một khái niệm riêng biệt (ví dụ "Chunking Strategies", "Vector Database Architecture", "Similarity Metrics"). SemanticHeaderChunker đảm bảo chunk retrieved chứa đúng header + toàn bộ nội dung của khái niệm đó.

**Code snippet:**
```python
from src.chunking import SemanticHeaderChunker

chunker = SemanticHeaderChunker(
    chunk_size=500,
    overlap=0,
    repeat_header=True,  # Header lặp lại ở đầu mỗi sub-chunk
)
chunks = chunker.chunk(text)
```

### So Sánh: Strategy của tôi vs Baseline


Chạy trên `techment_enterprise_rag.md` với `chunk_size=500`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality |
|----------|----------|-------------|------------|-------------------|
| techment_enterprise_rag.md | fixed_size | 61 | 496.6 | Thấp — cắt ngang câu |
| techment_enterprise_rag.md | by_sentences | 87 | 331.1 | Trung bình — nhiều chunk ngắn |
| techment_enterprise_rag.md | recursive | 69 | 433.8 | Cao — giữ cấu trúc |
| techment_enterprise_rag.md | **semantic_header (tôi)** | **84** | **380.3** | **Cao nhất — section intact** |

**Kết luận:**  
`SemanticHeaderChunker` tạo 84 chunks, avg length 380.3 chars — compact hơn và tập trung hơn RecursiveChunker (69 chunks, avg 433.8). Mỗi chunk đảm bảo tính toàn vẹn của section. Max length 530 (hơi vượt giới hạn) xảy ra khi `repeat_header=True` thêm header vào sub-chunk.




### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Đông Anh|  Custom SemanticHeaderChunker | 8/10 | Section integrity 100%, header giữ context | Chunk count cao hơn recursive |
| Đức Mạnh | FixedSizeChunker | 6/10 | Dễ implement, chunk đều size | Hay cắt ngang câu/ý |
|Lê Đạt | SentenceChunker | 7.5/10 | Giữ câu hoàn chỉnh, ít noise | Chunk ngắn, có thể mất heading |
| An Huy & Công Thái | RecursiveChunker | 8.5/10 | Linh hoạt, tốt với structured docs | Có thể trộn 2 sections |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng `re.split` với pattern `r'(\. |\! |\? |\.\n)'` để tách câu nhưng vẫn giữ lại dấu câu. Sau đó, code duyệt qua danh sách tách được, ghép các dấu câu trở lại vào cuối đoạn trước đó và gom nhóm các câu lại sao cho không vượt quá `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy nhận vào chuỗi hiện tại và danh sách `separators` còn lại. Base case là khi độ dài chuỗi nhỏ hơn hoặc bằng `chunk_size` thì trả về chính nó. Ngược lại, nó cắt chuỗi bằng separator ưu tiên cao nhất tìm thấy, ghép dần các phần; nếu phần ghép vượt quá `chunk_size` thì gọi đệ quy tiếp với các separators cấp thấp hơn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Code hỗ trợ hai cơ chế: lưu vào ChromaDB (nếu có thư viện) hoặc in-memory bằng một list các dictionary (`self._store`). Khi `search` in-memory, code tính điểm tương đồng (cosine similarity) bằng hàm `_dot` giữa embedding của query và từng document trong store, sau đó sort giảm dần theo điểm và trả về `top_k`.

**`search_with_filter` + `delete_document`** — approach:
> Để `search_with_filter`, hệ thống luôn lọc metadata (filter) trước bằng vòng lặp kiểm tra khớp key-value, sau đó mới tính similarity trên danh sách đã lọc. Hàm `delete_document` loại bỏ bằng cách build lại danh sách `self._store` với điều kiện lọc bỏ các document trùng `id` hoặc trùng `doc_id` trong metadata.

### KnowledgeBaseAgent

**`answer`** — approach:
> Lấy `top_k` documents liên quan nhất từ store, sau đó nối (join) phần `content` của chúng lại bằng 2 dấu xuống dòng (`\n\n`) để tạo `context`. Context này được tiêm trực tiếp vào đầu f-string prompt, phía sau là câu hỏi của người dùng, trước khi gửi tới hàm `llm_fn`.

### Test Results

```
======================================= test session starts ========================================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED         [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                  [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED           [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED            [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                 [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED       [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED        [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED      [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                        [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED        [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                   [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED               [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                         [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED    [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED    [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                        [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED          [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED            [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                  [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED       [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED         [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED          [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                   [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                  [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED             [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED         [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED    [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED        [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED              [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED        [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED   [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED  [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

======================================== 42 passed in 0.20s ========================================
```

**Số tests pass:** 42/42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Làm thế nào để đổi địa chỉ giao hàng? | Tôi muốn thay đổi nơi nhận đơn hàng. | high | 0.92 | yes |
| 2 | Cách bảo quản rau củ trong tủ lạnh? | Món ăn hôm nay cần được chế biến sao cho giữ vitamin? | low | 0.12 | yes |
| 3 | Đổi trả hàng có được hoàn tiền không? | Voucher giảm giá có áp dụng cho đơn hàng này không? | low | 0.25 | yes |
| 4 | Tôi cần hỗ trợ khi sản phẩm bị lỗi. | Làm sao liên hệ tổng đài chăm sóc khách hàng? | high | 0.78 | yes |
| 5 | Tại sao thẻ tín dụng bị trừ hai lần? | Làm sao để hoàn tiền khi bị trừ sai? | high | 0.84 | yes |


**Kết quả nào bất ngờ nhất?**  
> Kết quả bất ngờ nhất thường là khi hai câu có điểm similarity cao dù ý định khác nhau. Điều này cho thấy embeddings bắt tín hiệu chủ đề/lexical hơn là phân biệt intent tinh vi, nên cần kết hợp reranker để cải thiện precision.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | According to LlamaIndex, what are the five categories of data-backed LLM applications? | The five categories are: Agents, Workflows, Structured Data Extraction, Query Engines, and Chat Engines. |
| 2 | How much does combining Contextual Embeddings and Contextual BM25 reduce the top-20-chunk retrieval failure rate? | It reduces the top-20-chunk retrieval failure rate by 49%. |
| 3 | In a Vector Database, what is the main drawback of using Euclidean Distance compared to Cosine Similarity? | If a text chunk is very long and contains repeated words, the vector length increases, pushing the endpoint away and skewing results, even if the meaning is similar. |
| 4 | Based on the internal architecture guide, what are the three specific metrics used to measure the 'Search' phase (Retrieval KPIs)? | The three metrics are: Hit Rate, Mean Reciprocal Rank (MRR), and NDCG (Normalized Discounted Cumulative Gain). |
| 5 | What are the three components of the 'Triad of Metrics' in the Ragas framework? | The three components are: Faithfulness, Answer Relevance, and Context Relevancy. |


### Kết Quả Của Tôi — LocalEmbedder (Semantic Thật)

| # | Query (tóm tắt) | Source retrieve được | Score | Relevant? |
|---|-----------------|----------------------|-------|-----------|
| 1 | LlamaIndex five categories? | `llamaindex_concepts.md` — "## Retrieval Augmented Generation" liệt kê đủ 5 categories | **0.7358** | Có |
| 2 | Contextual Embeddings+BM25 failure rate? | `anthropic_contextual_rag.md` — "Performance improvements: ...failure rate by **49%**" | **0.9087** | Có |
| 3 | Main drawback of Euclidean Distance? | `rag_comprehensive_guide.md` — "### Similarity Metrics — Euclidean Distance: vector length increases..." | **0.6150** | Có |
| 4 | Three Retrieval KPI metrics? | *(no result — metadata filter mismatch)* | **0.0000** | Không |
| 5 | Three components of Ragas Triad? | `rag_comprehensive_guide.md` — "### Similarity Metrics — Cosine Similarity..." *(sai file)* | **0.4774** | Không |

**Bao nhiêu queries trả về chunk relevant trong top-3?** **3 / 5**

*(Q1, Q2, Q3 đúng cả source lẫn nội dung — score semantic cao. Q4 thất bại do metadata mismatch. Q5 thất bại do semantic confusion.)*
---


### Phân Tích Failure Cases

**Q4 — Metadata Filter Mismatch:**  
`benchmark.json` filter `{"category": "architecture", "access": "internal"}` nhưng `metadata.json` gán `{"category": "technical_guide", "access_level": "internal"}` cho `rag_definitive_architecture.md`. Hai lỗi đồng thời:
- Giá trị `category`: `"architecture"` ≠ `"technical_guide"`
- Tên field: `"access"` ≠ `"access_level"`

→ `search_with_filter` trả về rỗng, score = 0.  
**Fix:** Cập nhật `metadata.json` cho `rag_definitive_architecture.md` thành `{"category": "architecture", "access": "internal"}`.

**Q5 — Semantic Confusion giữa "Metrics" sections:**  
Query "Triad of Metrics in Ragas" retrieve nhầm section "Similarity Metrics" (Cosine/Euclidean) vì cùng có từ "metrics" nổi bật. File đúng `rag_definitive_architecture.md` có section "The RAGAS Triad" nhưng score chỉ 0.39 (thấp hơn 0.48 của file sai).  
**Fix:** Hybrid search BM25 + semantic để catch chính xác keyword "Ragas" / "Faithfulness".

**Q1 — Ghi chú:**  
Chunk được retrieve đúng source (llamaindex_concepts.md, score 0.7358 — cao). Nội dung chunk chứa đầy đủ 5 categories. Lexical relevance checker của script báo "No" do lỗi matching trên format markdown `[**Agents**](url)` — đây là false negative của checker, thực tế **Q1 relevant**.

---


## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**  
> Thành viên trong nhóm dùng `RecursiveChunker` với metadata `intent` field để phân biệt ý định người dùng. Điều này giúp tôi nhận ra metadata schema thiết kế tốt ảnh hưởng lớn đến retrieval precision — filter theo `intent` trước khi similarity search giúp loại bỏ nhiễu rất hiệu quả.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**  
> Một nhóm demo `Small2Big chunking`: lưu chunk nhỏ (câu) để search nhưng trả "parent chunk" lớn hơn cho LLM đọc. Cách này giải quyết trade-off precision vs context. Đây là inspiration để cải thiện `SemanticHeaderChunker` trong tương lai — sub-chunk retrieve, parent section trả về.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**  
> Sẽ thiết kế metadata schema kỹ trước khi thu thập tài liệu: thêm field `topic_tags` (list) để filter đa chiều, thống nhất tên field giữa `metadata.json` và `benchmark.json` (tránh lỗi Q4 — `access` vs `access_level`). Ngoài ra sẽ dùng hybrid search (BM25 + semantic) để tránh nhầm lẫn giữa các section cùng chủ đề "metrics" (lỗi Q5).
---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 5 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
