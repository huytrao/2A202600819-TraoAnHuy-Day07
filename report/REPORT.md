# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Công Thái
**Nhóm:** B4
**Ngày:** 06/05/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**  
High cosine similarity nghĩa là hai vector embedding có hướng gần giống nhau, tức là hai câu/tài liệu có ý nghĩa ngữ nghĩa gần nhau dù không nhất thiết dùng cùng từ. Giá trị càng gần `1` thì mức độ tương đồng về nghĩa càng cao.

**Ví dụ HIGH similarity:**
- Sentence A: Tôi muốn đổi địa chỉ giao hàng sau khi đã đặt món.
- Sentence B: Làm sao để thay đổi nơi nhận đơn sau khi đặt đồ ăn?
- Tại sao tương đồng: Hai câu dùng từ khác nhau nhưng cùng hỏi về việc thay đổi địa chỉ giao hàng sau khi đã đặt đơn.

**Ví dụ LOW similarity:**
- Sentence A: Tôi muốn đổi địa chỉ giao hàng sau khi đã đặt món.
- Sentence B: Cách bảo quản rau củ trong tủ lạnh được lâu hơn là gì?
- Tại sao khác: Hai câu thuộc hai ý định khác nhau: một câu hỏi về xử lý đơn hàng, một câu hỏi về bảo quản thực phẩm.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**  
Cosine similarity tập trung vào hướng của vector nên phù hợp để đo mức độ giống nhau về nghĩa, ít bị ảnh hưởng bởi độ dài hoặc độ lớn tuyệt đối của embedding. Với text embeddings, hai câu có thể có vector độ lớn khác nhau nhưng vẫn cùng hướng ngữ nghĩa, nên cosine thường ổn định hơn Euclidean distance.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, `chunk_size=500`, `overlap=50`. Bao nhiêu chunks?**  

> *Trình bày phép tính:*  
> `stride = chunk_size - overlap = 500 - 50 = 450`  
> Số chunk cần để bao phủ document:  
> `n = ceil((10000 - 500) / 450) + 1`  
> `n = ceil(9500 / 450) + 1 = ceil(21.11) + 1 = 22 + 1 = 23`

> *Đáp án:* **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**  
Khi `overlap=100`, `stride = 500 - 100 = 400`, nên `n = ceil((10000 - 500) / 400) + 1 = ceil(23.75) + 1 = 25 chunks`. Overlap nhiều hơn làm số chunk tăng, nhưng giúp giữ ngữ cảnh giữa hai chunk liên tiếp tốt hơn, giảm nguy cơ một ý quan trọng bị cắt đôi.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn



### Data Inventory


### Metadata Schema


---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis


### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**  
Strategy của tôi tách tài liệu theo thứ tự ưu tiên từ cấu trúc lớn đến nhỏ: heading markdown, đoạn văn, câu, rồi cuối cùng mới cắt theo kích thước ký tự nếu đoạn vẫn quá dài. Cách này giúp chunk giữ được ngữ cảnh tự nhiên của tài liệu FAQ, ví dụ heading “Điều kiện hoàn tiền” sẽ đi cùng các câu giải thích bên dưới. Nếu một đoạn vượt quá `chunk_size`, thuật toán tiếp tục split bằng separator nhỏ hơn cho đến khi chunk đủ ngắn. Tôi vẫn dùng overlap nhỏ để chunk sau có một phần ngữ cảnh từ chunk trước.

**Tại sao tôi chọn strategy này cho domain nhóm?**  
Domain FAQ thường có cấu trúc rõ ràng: tiêu đề vấn đề, điều kiện, các bước xử lý và lưu ý. RecursiveChunker khai thác tốt cấu trúc này vì nó không cắt máy móc theo số ký tự như FixedSizeChunker. Nhờ vậy, khi user hỏi một vấn đề cụ thể như “voucher không dùng được”, retrieval có khả năng lấy đúng cả nguyên nhân và cách xử lý trong cùng một chunk.

**Code snippet (nếu custom):**
```python
# Tôi không dùng custom strategy.
# Strategy chính: RecursiveChunker có sẵn trong package src.
# Cấu hình đề xuất:
chunker = RecursiveChunker(
    chunk_size=500,
    overlap=50,
    separators=["\n## ", "\n### ", "\n\n", ". ", " ", ""]
)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |



### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|


**Strategy nào tốt nhất cho domain này? Tại sao?**  


---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng một biểu thức chính quy tách câu theo các dấu kết thúc câu như '.', '?', '!' và '…' (kết hợp lookahead/lookbehind để xác định ranh giới câu) và loại trừ các trường hợp không phải kết thúc câu như chữ viết tắt (ví dụ 'Mr.', 'i.e.'), số thập phân và URL/email.
> Sau khi tách, strip khoảng trắng và gộp những câu quá ngắn khi cần để đạt `chunk_size` tối thiểu, đồng thời giữ `overlap` giữa các chunk để bảo toàn ngữ cảnh.
> Các edge case cần xử lý bao gồm chữ viết tắt, số thập phân, URL/email, bullet points/danh sách, và văn bản thiếu dấu câu rõ ràng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> RecursiveChunker hoạt động theo kiểu đệ quy: nó thử tách đoạn theo danh sách separator từ lớn đến nhỏ (ví dụ heading → đoạn văn → câu) để giữ ngữ cảnh tự nhiên, và nếu một phần con vẫn vượt quá `chunk_size` thì gọi `_split` đệ quy trên phần đó với separator tiếp theo.
> Base case là khi đoạn nhỏ hơn hoặc bằng `chunk_size`, hoặc khi không còn separator nào để tách — khi đó trả về đoạn hiện tại (có thể trim và điều chỉnh overlap).
> Thuật toán cũng đảm bảo giữ `overlap` giữa các chunk và gộp các phần quá ngắn nếu cần để tránh tạo nhiều chunk không có ngữ nghĩa.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Khi `add_documents`, mỗi document/chunk được mã hóa thành embedding bằng một model embedding rồi lưu kèm `id`, `text` và `metadata` trong store (có thể là list, DB hoặc ANN index như FAISS/HNSW); vectors nên được chuẩn hóa (unit norm) để đơn giản hóa phép tính similarity.
> Khi `search`, sinh embedding cho query rồi tính cosine similarity giữa query vector và các vectors trong store (hoặc truy vấn ANN index để tăng tốc), sau đó trả về top-k kết quả cùng score và metadata.
> Lưu ý về scale: dùng chỉ mục ANN để tăng tốc trên tập lớn và lưu trữ metadata riêng để cho phép lọc/hiển thị nhanh.

**`search_with_filter` + `delete_document`** — approach:
> Nên áp dụng filter theo metadata trước khi tính similarity để thu hẹp candidate set (filter-first), điều này giảm chi phí truy vấn ANN/full-scan và đảm bảo chỉ so sánh với các document thỏa điều kiện.
> Để xóa document, loại bỏ hoặc đánh dấu `id` trong store và metadata; với ANN index, nếu xóa trực tiếp không được hỗ trợ thì đánh dấu tombstone và rebuild/refresh index định kỳ để loại bỏ các mục đã xóa.
> Đồng thời xóa metadata và nội dung lưu trữ (DB hoặc file) để giải phóng dung lượng và giữ nhất quán giữa index và store.

### KnowledgeBaseAgent


**`answer`** — approach:
> Prompt nên gồm ba phần: `system` instruction ngắn mô tả nhiệm vụ và phong cách trả lời, một phần `Context` chứa top-k chunks truy xuất (mỗi chunk kèm id/metadata), và cuối cùng là `user` query.
> Khi inject context, chèn các chunk đã được dán nhãn trước câu hỏi và rõ ràng hướng dẫn model “use only the provided context to answer; if insufficient, respond with 'I don't know'”, đồng thời yêu cầu trích dẫn nguồn metadata cho các thông tin cụ thể để giảm hallucination.
> Nếu context quá dài, tóm tắt hoặc cắt bớt theo token budget trước khi inject, và dùng temperature thấp cùng kiểm soát length để đảm bảo câu trả lời ngắn gọn, chính xác.

### Test Results

```
# Paste output of: pytest tests/ -v
```
======================================= test session starts ========================================
platform win32 -- Python 3.12.10, pytest-9.0.3, pluggy-1.6.0 -- C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\admin\Day-07-Lab-Data-Foundations
plugins: anyio-4.13.0, langsmith-0.8.9, asyncio-1.4.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
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
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED    [ 38%]
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

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Làm thế nào để đổi địa chỉ giao hàng? | Tôi muốn thay đổi nơi nhận đơn hàng. | high | 0.92 | yes |
| 2 | Cách bảo quản rau củ trong tủ lạnh? | Món ăn hôm nay cần được chế biến sao cho giữ vitamin? | low | 0.12 | yes |
| 3 | Đổi trả hàng có được hoàn tiền không? | Voucher giảm giá có áp dụng cho đơn hàng này không? | low | 0.25 | yes |
| 4 | Tôi cần hỗ trợ khi sản phẩm bị lỗi. | Làm sao liên hệ tổng đài chăm sóc khách hàng? | high | 0.78 | yes |
| 5 | Tại sao thẻ tín dụng bị trừ hai lần? | Làm sao để hoàn tiền khi bị trừ sai? | high | 0.84 | yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất thường là khi hai câu có điểm similarity cao dù ý định khác nhau — nguyên nhân phổ biến là chúng chia sẻ nhiều từ khóa hoặc chủ đề chung (ví dụ cùng nhắc tới sản phẩm hoặc quy định). Điều này cho thấy embeddings ưu tiên bắt các tín hiệu chủ đề/lexical hơn là phân biệt ý định tinh vi, nên để cải thiện độ chính xác về intent nên kết hợp thêm reranker hoặc các tín hiệu ngữ cảnh bổ sung.


---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |

**Bao nhiêu queries trả về chunk relevant trong top-3?** __ / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:*

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
