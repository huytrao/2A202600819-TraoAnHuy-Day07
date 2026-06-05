# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Hữu Đạt
**Nhóm:** B4
**Ngày:** 6/5/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* High cosine similarity nghĩa là hai vector nằm gần nhau trong không gian nhiều chiều, cho thấy chúng có hướng và ý nghĩa tương tự nhau. Điều này có nghĩa là 
sample dữ liệu có độ tương đồng ngữ nghĩa cao.

**Ví dụ HIGH similarity:**
- Sentence A: The cat sat on the mat.
- Sentence B: The feline rested upon the rug.
- Tại sao tương đồng: Cả hai câu đều diễn tả hành động một con mèo ngồi lên tấm thảm.

**Ví dụ LOW similarity:**
- Sentence A: How old are you?
- Sentence B: I am 20 years old.
- Tại sao khác: Câu A hỏi về tuổi của đối tượng được nhắc đến, trong khi câu B trả lời về tuổi của người nói. Mặc dù cả hai câu đều liên quan đến chủ đề tuổi tác, nhưng chúng có ý nghĩa và mục đích hoàn toàn khác nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Cosine similarity không phụ thuộc vào độ dài của vector, mà chỉ quan tâm đến góc giữa chúng. Điều này có nghĩa là khi tăng chiều hay giảm chiều không gian thì cosine similarity vẫn giữ nguyên giá trị còn Euclidean distance thì thay đổi theo không gian.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Áp dụng công thức: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Khi overlap tăng lên 100, bước nhảy giữa các chunk giảm xuống (còn 400), dẫn đến số lượng chunk **tăng lên** (thành 25 chunks). Ta muốn overlap nhiều hơn để tránh trường hợp các câu hoặc ý nghĩa bị cắt ngang giữa chừng, giúp bảo toàn ngữ cảnh tốt hơn khi retrieval.

---


## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** RAG technical documents

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:* Domain này tập trung vào tài liệu kỹ thuật về Retrieval-Augmented Generation (RAG) và Vector Store — đây là nền tảng cốt lõi của Lab Day 7. Các tài liệu có cấu trúc markdown rõ ràng với header phân cấp, phù hợp để so sánh các chunking strategy khác nhau. Nội dung đủ phức tạp (30K ký tự) để thể hiện sự khác biệt rõ rệt giữa các strategy, đồng thời có thể thiết kế benchmark queries đòi hỏi thông tin cụ thể từ đúng tài liệu.

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
| category | string | business_blog, documentation | Giúp lọc nhanh tài liệu theo loại nội dung (bài báo, tài liệu kỹ thuật, ghi chú) trước khi search. |
| access_level | string | public, internal | Kiểm soát quyền truy cập, đảm bảo không trả về tài liệu nội bộ (internal) cho người dùng public. |
| source | string | LlamaIndex Docs | Dễ dàng trích xuất nguồn gốc thông tin để tăng tính minh bạch cho câu trả lời của AI. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| llamaindex_concepts.md | FixedSizeChunker (`fixed_size`) | 21 | 290.90 | Trung bình (có thể bị cắt ngang câu) |
| llamaindex_concepts.md | SentenceChunker (`by_sentences`) | 11 | 517.00 | Tốt (giữ nguyên câu trọn vẹn) |
| llamaindex_concepts.md | RecursiveChunker (`recursive`) | 28 | 202.07 | Rất tốt (chia tách theo đoạn văn/câu hợp lý) |

### Strategy Của Tôi

**Loại:** SentenceChunker (by_sentences)

**Mô tả cách hoạt động:**
> *Viết 3-4 câu:* Strategy này không cắt văn bản một cách thô bạo theo số lượng ký tự cố định, mà sử dụng Regex (biểu thức chính quy) với Lookbehind để tìm các dấu hiệu kết thúc câu như dấu chấm, chấm hỏi, chấm than, hoặc xuống dòng. Sau khi nhận diện được danh sách các câu trọn vẹn, thuật toán sẽ gộp (group) một số lượng câu nhất định lại với nhau (ví dụ: `max_sentences_per_chunk=3`) nhằm tạo ra các chunk có ý nghĩa ngữ nghĩa hoàn chỉnh, không bị gãy cấu trúc.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu:* Tài liệu kỹ thuật, blog hướng dẫn và tài liệu RAG thường dùng cấu trúc diễn đạt thành các câu hoàn chỉnh để giải thích khái niệm phức tạp. Việc phân tách dựa trên câu sẽ giúp giữ nguyên được ý tưởng trọn vẹn của tác giả, đảm bảo mô hình LLM khi đọc được context sẽ không bị mất nửa câu hoặc nhận được nửa từ ngữ vô nghĩa như cách chia cố định số lượng ký tự.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Đông Anh|  Custom SemanticHeaderChunker | 8/10 | Section integrity 100%, header giữ context | Chunk count cao hơn recursive |
| Đức Mạnh | FixedSizeChunker | 6/10 | Dễ implement, chunk đều size | Hay cắt ngang câu/ý |
|Lê Đạt | SentenceChunker | 7.5/10 | Giữ câu hoàn chỉnh, ít noise | Chunk ngắn, có thể mất heading |
| An Huy & Công Thái | RecursiveChunker | 8.5/10 | Linh hoạt, tốt với structured docs | Có thể trộn 2 sections | 
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
> *Viết 2-3 câu:* Dùng regex `(?<=[.!?])\s+|(?<=\.\n)` kết hợp Lookbehind để nhận diện ranh giới câu mà không làm mất dấu kết thúc câu gốc. Sau đó gom nhiều câu liên tiếp lại thành một chunk cho đến khi đạt ngưỡng `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu:* Đệ quy chia văn bản dựa trên mảng separators (ví dụ `["\n\n", "\n", ". "]`). Base case là khi chuỗi hiện tại đã nhỏ hơn `chunk_size` hoặc khi hết list separators. Sau đó duyệt qua các chuỗi được cắt, nếu chuỗi nhỏ thì gom lại, nếu gộp vào mà vượt quá thì mới chia đôi hoặc đẩy xuống level separator tiếp theo.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu:* Chuyển đổi `Document` content thành embeddings thông qua `embedding_fn`. Khi search, tính embedding của câu hỏi, sau đó quét qua toàn bộ docs trong memory và tính cosine similarity rồi sort giảm dần để lấy top_k documents cao điểm nhất.

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu:* Lọc `metadata` trước bằng cách duyệt xem document có khớp toàn bộ key-value trong bộ filter không, loại bớt ứng viên rồi mới tiến hành tính cosine similarity (tối ưu tính toán). Khi delete, tạo lại mảng documents chỉ chứa các document không trùng ID cần xoá.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu:* Thực hiện retrieval qua hàm `search` lấy top_k chunks. Sau đó format các chunks thành một chuỗi context tổng hợp và nhúng vào Prompt Template cùng câu hỏi để truyền vào LLM sinh ra câu trả lời theo ngữ cảnh.

### Test Results

```
19 passed in 0.54s
```

**Số tests pass:** __ / __

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Retrieval-Augmented Generation helps LLMs access external knowledge. | RAG systems allow language models to retrieve outside information. | high | 0.3762 | (tuỳ) |
| 2 | Vector databases store embeddings for similarity search. | Chunking text into smaller pieces is crucial for good RAG performance. | low | 0.0474 | (tuỳ) |
| 3 | Retrieval-Augmented Generation helps LLMs access external knowledge. | I love eating pizza with extra cheese on weekends. | low | 0.0239 | (tuỳ) |
| 4 | Python is a great language for beginners. | Python is a terrible language for beginners. | low | 0.8495 | Sai |
| 5 | Tôi thích uống cà phê đen đá không đường. | Ly cà phê đen đá không đường là thức uống yêu thích của tôi. | high | 0.8656 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:* Kết quả của Pair 4 (Python is great vs terrible) bất ngờ nhất vì dù có ý nghĩa trái ngược nhau hoàn toàn, actual score lại cực kỳ cao (0.8495). Điều này cho thấy embeddings phụ thuộc nhiều vào ngữ cảnh từ vựng; hai câu có cấu trúc và các từ giống nhau gần như y hệt sẽ có vector rất gần nhau trong không gian, dù có một từ mang ý nghĩa phủ định/ngược nghĩa.

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

### Kết Quả Của Tôi (Dựa trên Sentence-based Strategy)

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | According to LlamaIndex... | "You can learn more about RAG... There are endless use cases for data-backed LLM applications..." | 0.633 | Yes | [Answer generated by LLM] |
| 2 | How much does combining Contextual... | "Combining Contextual Embeddings and Contextual BM25 reduced the top-20-chunk retrieval failure rate by 49%" | 0.860 | Yes | [Answer generated by LLM] |
| 3 | In a Vector Database... | "2. **Euclidean Distance ($L_2$ Distance):** Measures the absolute straight-line distance between vector endpoints..." | 0.615 | Yes | [Answer generated by LLM] |
| 4 | Based on the internal architecture guide... | Không tìm thấy kết quả relevant trong tập dữ liệu | N/A | No | [Answer generated by LLM] |
| 5 | What are the three components... | "Related Insights: Learn how our Microsoft Fabric Readiness Assessment explores..." | 0.365 | No | [Answer generated by LLM] |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 3 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:* học được cách làm việc nhóm, phối hợp với mọi người. học được cách giải quyết mâu thuẫn công việc.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:* dựa theo từng tính chất của loại dữ liệu thì sẽ có cách handle và chunking khác nhau để tạo dữ liệu

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:* với dữ liệu technical thì sẽ format nội dung thành các section sau đó mới tiến hành chunking thì sẽ tốt hơn. từ đó xây dựng structure data để hỗ trợ cho retrieval tốt hơn.
---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5/ 5 |
| Document selection | Nhóm | 9/ 10 |
| Chunking strategy | Nhóm | 15/ 15 |
| My approach | Cá nhân | 9/ 10 |
| Similarity predictions | Cá nhân | 5/ 5 |
| Results | Cá nhân | 8/ 10 |
| Core implementation (tests) | Cá nhân | 25/ 30 |
| Demo | Nhóm | 5/ 5 |
| **Tổng** | | **91/ 100** |
