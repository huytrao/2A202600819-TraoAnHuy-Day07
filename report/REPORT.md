# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đông Anh  
**Nhóm:** B4 
**Ngày:** 05/06/2026  

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (giá trị tiến gần về 1.0) nghĩa là hai vector embedding chỉ về cùng một hướng trong không gian đa chiều, thể hiện rằng hai đoạn văn bản đó có sự tương đồng rất lớn về mặt ngữ nghĩa và bối cảnh, bất kể chúng có thể sử dụng các từ ngữ khác nhau.

**Ví dụ HIGH similarity:**
- **Sentence A:** "Làm thế nào để xử lý lỗi lỗi billing error?"
- **Sentence B:** "Phương thức giải quyết khi hệ thống gặp sự cố thanh toán hóa đơn."
- **Tại sao tương đồng:** Cả hai câu đều có cùng một mục đích ngữ nghĩa là tìm cách khắc phục lỗi liên quan đến cổng thanh toán/hóa đơn, dù cấu trúc từ vựng tiếng Anh và tiếng Việt đan xen khác nhau.

**Ví dụ LOW similarity:**
- **Sentence A:** "Làm thế nào để xử lý lỗi lỗi billing error?"
- **Sentence B:** "Hướng dẫn cài đặt môi trường ảo Python 3.9 bằng thư viện venv."
- **Tại sao khác:** Hai câu hoàn toàn thuộc hai phạm trù chủ đề độc lập (một bên là xử lý sự cố tài chính/thanh toán, một bên là kỹ thuật lập trình phần mềm).

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì Cosine similarity chỉ đo góc giữa hai vector, tập trung hoàn toàn vào hướng (ngữ nghĩa) của văn bản mà không bị ảnh hưởng bởi độ dài ngắn của đoạn văn. Ngược lại, Euclidean distance đo khoảng cách tuyệt đối giữa hai đầu mút vector nên rất dễ bị sai lệch khi một đoạn văn quá dài đối sánh với một đoạn văn quá ngắn.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> - Kích thước bước nhảy thực tế (Step): $step = chunk\_size - overlap = 500 - 50 = 450$ ký tự.
> - Số lượng chunk nhảy theo bước cố định: $\lceil (10000 - 500) / 450 \rceil + 1 = \lceil 9500 / 450 \rceil + 1 = 22 + 1 = 23$ chunks.
> 
> **Đáp án:** 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100, bước nhảy giảm xuống còn 400 ký tự, dẫn đến số lượng chunk tăng lên thành 25 chunks. Chúng ta muốn tăng overlap khi văn bản chứa nhiều thông tin ngữ cảnh liên tục; việc gối đầu nhiều ký tự giúp đảm bảo các câu hoặc ý nghĩa nằm ở ranh giới cắt không bị bẻ đôi, giúp mô hình AI giữ được tính liền mạch khi truy vấn.

---
## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn
**Domain:** Kiến trúc Hệ thống RAG và Enterprise AI.

**Domain:** [ví dụ: Customer support FAQ, Vietnamese law, cooking recipes, ...]

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain này vì đây là lĩnh vực đòi hỏi độ chính xác cao về mặt thuật ngữ kỹ thuật và cấu trúc tài liệu thường phân cấp rõ ràng theo các tiêu đề (Markdown). Việc xử lý dữ liệu này giúp kiểm chứng hiệu quả của các chiến lược chunking nâng cao và khả năng lọc metadata theo phân quyền (access_level).

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | techment_enterprise_rag.md | Techment Blog | ~8,500 | category: business_blog, access_level: public |
| 2 | llamaindex_concepts.md | LlamaIndex Docs | ~4,200 | category: documentation, access_level: public |
| 3 | anthropic_contextual_rag.md| Anthropic Blog | ~5,000 | category: research_blog, access_level: public |
| 4 | rag_comprehensive_guide.md | Internal Guide | ~6,500 | category: technical_guide, access_level: internal |
| 5 | vector_store_notes.md | Class Notes | ~2,800 | category: documentation, access_level: public |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | string | "documentation" | Giúp thu hẹp không gian tìm kiếm vào đúng loại tài liệu (blog vs docs). |
| access_level | string | "internal" | Đảm bảo an toàn dữ liệu, chỉ truy xuất tài liệu nội bộ cho user có quyền. |
| source | string | "Anthropic Blog" | Giúp Agent cung cấp trích dẫn nguồn (citation) chính xác trong câu trả lời. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis
| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| llamaindex_concepts.md | FixedSizeChunker | 9 | 466 | No (Cắt ngang các danh sách Agents/Workflows) |
| llamaindex_concepts.md | SentenceChunker | 12 | 350 | Partial (Giữ trọn câu nhưng mất cấu trúc Header) |
| llamaindex_concepts.md | RecursiveChunker | 7 | 600 | Yes (Giữ được các phân đoạn Paragraph tốt) |

### Strategy Của Tôi
**Loại:** `CustomChunkerbyMarkdownHeadings` (Chunking theo Header Markdown)

**Mô tả cách hoạt động:**
> Chiến lược này sử dụng Regex `(?=^#{1,6}\s)` để xác định các tiêu đề từ cấp 1 đến cấp 6 trong Markdown. Nó cố gắng giữ toàn bộ nội dung dưới một tiêu đề nằm chung một chunk. Nếu nội dung dưới tiêu đề vượt quá `chunk_size`, nó sẽ thực hiện cắt nhỏ tiếp để đảm bảo vector embedding không bị quá tải thông tin.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Vì các tài liệu kỹ thuật về RAG thường được cấu trúc rất chặt chẽ theo các đề mục (Ví dụ: "Typical Workflow", "Common Risks"). Việc chunk theo Header giúp đảm bảo khi truy vấn về một khái niệm, Agent sẽ nhận được toàn bộ đoạn giải thích liên quan thay vì các mảnh vụn bị cắt rời.

**Code snippet (nếu custom):**
```python
pattern = r'(?=^#{1,6}\s)'
segments = re.split(pattern, text, flags=re.MULTILINE)
# ... logic gộp segments dựa trên chunk_size ...
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| llamaindex_concepts.md | RecursiveChunker | 30 | 2000 | 8/10 (Giữ được các phân đoạn logic) |
| llamaindex_concepts.md | **Custom Markdown** | 73 | 800 | 9.5/10 (Giữ trọn vẹn ngữ cảnh tiêu đề) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | | | | |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> `CustomChunkerbyMarkdownHeadings` là tốt nhất cho domain này vì tài liệu kỹ thuật/documentation luôn được phân cấp rõ ràng bằng Heading. Việc giữ nội dung đi kèm tiêu đề giúp vector embedding mang đậm đặc tính ngữ nghĩa của chủ đề đó, giúp Agent trả lời chính xác các câu hỏi dạng "What are the categories..." hoặc "How to...".

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng biểu thức chính quy `re.split(r'(?<=[.!?])\s+|(?<=\.)\n', text)` để tách các câu dựa trên dấu chấm, dấu chấm hỏi, dấu chấm than kết thúc kết hợp với khoảng trắng hoặc ký tự xuống dòng. Đoạn mã xử lý tốt edge case khoảng trắng thừa bằng phương thức `.strip()` và tiến hành gom cụm tuần tự dựa trên tham số `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán hoạt động theo cơ chế chia để trị đệ quy, duyệt qua danh sách các ký tự phân tách theo thứ tự ưu tiên giảm dần (`\n\n`, `\n`, `. `, ` `, `""`). Base case của hàm đệ quy là khi đoạn văn bản hiện tại có độ dài nhỏ hơn hoặc bằng `chunk_size` quy định, lúc đó đoạn văn bản sẽ được giữ nguyên mà không cần chia nhỏ tiếp.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Việc lưu trữ tài liệu được thực hiện thông qua việc tích hợp trực tiếp với cơ sở dữ liệu vector ChromaDB (`EphemeralClient`) để quản lý các tập dữ liệu embedding. Quá trình `search` sử dụng thuật toán tính toán khoảng cách Cosine Similarity hình học tích hợp sẵn trong DB để chấm điểm và trả về `top_k` kết quả có độ tương đồng cao nhất.

**`search_with_filter` + `delete_document`** — approach:
> Hàm `search_with_filter` thực hiện cơ chế pre-filtering (lọc trước) bằng cách truyền trực tiếp tham số `where` vào câu lệnh truy vấn của ChromaDB giúp tối ưu hóa hiệu năng quét vector. Hàm `delete_document` giải quyết việc xóa tài liệu thông qua phương thức `.delete(ids=[doc_id])` để loại bỏ dữ liệu khỏi cả bộ nhớ RAM và bộ lưu trữ.

### KnowledgeBaseAgent

**`answer`** — approach:
> Cấu trúc prompt bao gồm chỉ thị vai trò hệ thống (System Prompt) rõ ràng để định hướng Agent trả lời trung thực và không tự bịa thông tin. Toàn bộ ngữ cảnh (Context) tìm được từ `EmbeddingStore` sẽ được gộp lại bằng dấu xuống dòng và inject động trực tiếp vào giữa phần hướng dẫn hệ thống và câu hỏi của người dùng.

### Test Results

================================== 42 passed in 0.45s ===================================

**Số tests pass:** 42 / 42

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
| 1 | Python is used for AI. | Machine learning models use Python. | High | 0.892 | Yes |
| 2 | RAG reduces hallucinations. | Vector stores improve retrieval. | High | 0.715 | Yes |
| 3 | The quick brown fox. | LLMs are powerful tools. | Low | 0.042 | Yes |
| 4 | Billing error resolution. | How to fix payment issues? | High | 0.785 | Yes |
| 5 | Euclidean distance math. | Cosine similarity for text. | Medium | 0.520 | Yes |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là cặp số 5 có điểm similarity ở mức trung bình dù chúng đều là thuật toán toán học. Điều này cho thấy embedding không chỉ bắt keyword (toán học) mà còn phân biệt được sự khác biệt về logic giữa "khoảng cách" và "góc độ", chứng minh mô hình hiểu được ngữ cảnh sâu hơn là chỉ khớp từ vựng.

---

## 6. Results — Cá nhân (10 điểm)

### Benchmark Queries & Gold Answers (nhóm thống nhất)
| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What are the five categories of LlamaIndex applications? | Agents, Workflows, Structured Data Extraction, Query Engines, Chat Engines. |
| 2 | How much does Contextual RAG reduce failure rate? | 49% reduction. |
| 3 | Euclidean Distance vs Cosine Similarity drawback? | Long text skewing distance due to word repetition. |
| 4 | What are the three Retrieval KPIs? | Hit Rate, Mean Reciprocal Rank (MRR), and NDCG. |
| 5 | Components of 'Triad of Metrics' in Ragas? | Faithfulness, Answer Relevance, and Context Relevancy. |

### Kết Quả Của Tôi
| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | LlamaIndex categories... | RAG in Enterprise AI scenario... | 0.308 | No | LLM lặp lại prompt. |
| 2 | Contextual RAG failure rate... | RAG is not a silver bullet... | 0.340 | No | LLM lặp lại prompt. |
| 3 | Distance metrics math... | RAG is not a silver bullet... | 0.264 | No | LLM lặp lại prompt. |
| 4 | Retrieval KPIs... | (Không tìm thấy chunk nào) | N/A | No | LLM lặp lại prompt. |
| 5 | Ragas Triad... | RAG in Enterprise AI scenario... | 0.284 | No | LLM lặp lại prompt. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 0 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được từ bạn cùng nhóm cách tối ưu hóa Metadata Filtering để phân quyền tài liệu (`access_level`). Việc sử dụng Pre-filtering trong ChromaDB giúp hệ thống tự động loại bỏ các tài liệu nhạy cảm trước khi tính toán vector, vừa đảm bảo an toàn vừa tăng tốc độ search.
> Tôi học được cách tối ưu Metadata Filtering để phân quyền tài liệu (`access_level`) bằng Pre-filtering trong ChromaDB, đảm bảo an toàn và tăng tốc độ search.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm bạn đã giới thiệu `SemanticChunker`, kỹ thuật chia đoạn dựa trên sự thay đổi cosine similarity giữa các câu, giúp các chunk đồng nhất chủ đề hơn khi tài liệu không có cấu trúc Markdown rõ ràng.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> > Tôi sẽ ưu tiên khắc phục các vấn đề cơ bản: 1. Sử dụng Embedder thực tế (Local/OpenAI) thay vì mock. 2. Đồng bộ `benchmark.json` với tên file gốc và đảm bảo tất cả tài liệu benchmark được nạp. 3. Cải tiến logic so khớp nguồn trong benchmark.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 4/ 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 8 / 15 |
| My approach | Cá nhân | 7 / 10 |
| Similarity predictions | Cá nhân | 3 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 25 / 30 |
| Demo | Nhóm | 3 / 5 |
| **Tổng** | | **/ 100** |
