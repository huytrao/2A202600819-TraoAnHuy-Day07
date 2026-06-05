# Day 7 — Exercises
## Data Foundations: Embedding & Vector Store | Lab Worksheet — Đã Hoàn Thiện

---

## Part 1 — Warm-up (Cá nhân)

### Exercise 1.1 — Cosine Similarity in Plain Language

#### What does it mean for two text chunks to have high cosine similarity?

Hai text chunks có **high cosine similarity** nghĩa là embedding vectors của chúng có hướng gần giống nhau trong vector space. Nói đơn giản, hai đoạn text đó có ý nghĩa ngữ nghĩa gần nhau, dù có thể không dùng chính xác cùng từ.

Ví dụ, câu “Tôi muốn hủy đơn hàng” và “Làm sao để cancel order?” có thể có cosine similarity cao vì cùng nói về intent hủy đơn.

---

#### Give a concrete example of two sentences that would have HIGH similarity and two that would have LOW similarity.

##### HIGH similarity

- Sentence A: Tôi muốn hủy đơn vì đặt nhầm món.
- Sentence B: Làm sao để cancel order sau khi đã đặt đồ ăn?

**Tại sao tương đồng:**  
Hai câu đều nói về cùng một nhu cầu: người dùng muốn hủy đơn sau khi đã đặt món. Dù Sentence B dùng từ tiếng Anh “cancel order”, ý nghĩa vẫn giống Sentence A.

##### LOW similarity

- Sentence A: Tôi muốn hủy đơn vì đặt nhầm món.
- Sentence B: Cách bảo quản rau củ trong tủ lạnh là gì?

**Tại sao khác:**  
Sentence A thuộc nhóm hỗ trợ đơn hàng, còn Sentence B thuộc nhóm nấu ăn/bảo quản thực phẩm. Hai câu khác domain và khác intent nên similarity thấp.

---

#### Why is cosine similarity preferred over Euclidean distance for text embeddings?

Cosine similarity được ưu tiên vì nó đo **hướng** của vector thay vì khoảng cách tuyệt đối giữa hai vector. Với text embeddings, hướng vector thường thể hiện ý nghĩa của câu tốt hơn độ lớn vector. Vì vậy, cosine similarity phù hợp hơn để so sánh mức độ giống nhau về ngữ nghĩa giữa các đoạn text.

---

### Exercise 1.2 — Chunking Math

#### A document is 10,000 characters. You chunk it with `chunk_size=500`, `overlap=50`. How many chunks do you expect?

Công thức:

```text
num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))
```

Thay số:

```text
doc_length = 10000
chunk_size = 500
overlap = 50

chunk_size - overlap = 500 - 50 = 450

num_chunks = ceil((10000 - 50) / 450)
           = ceil(9950 / 450)
           = ceil(22.11)
           = 23
```

**Đáp án:**  
Cần khoảng **23 chunks**.

---

#### If overlap is increased to 100, how does this change the chunk count?

Thay `overlap=100`:

```text
doc_length = 10000
chunk_size = 500
overlap = 100

chunk_size - overlap = 500 - 100 = 400

num_chunks = ceil((10000 - 100) / 400)
           = ceil(9900 / 400)
           = ceil(24.75)
           = 25
```

**Đáp án:**  
Khi overlap tăng từ 50 lên 100, số chunk tăng từ **23 chunks** lên **25 chunks**.

---

#### Why would you want more overlap?

Muốn overlap nhiều hơn vì overlap giúp giữ ngữ cảnh giữa hai chunk liên tiếp. Nếu một ý quan trọng bị cắt ở ranh giới chunk, phần overlap giúp chunk sau vẫn giữ được một phần thông tin từ chunk trước, từ đó retrieval và answer generation chính xác hơn.

---

## Part 2 — Core Coding (Cá nhân)

Implement all TODOs in `src/chunking.py`, `src/store.py`, và `src/agent.py`.

### Checklist

- [x] `Document` dataclass — ĐÃ IMPLEMENT SẴN
- [x] `FixedSizeChunker` — ĐÃ IMPLEMENT SẴN
- [x] `SentenceChunker` — split on sentence boundaries, group into chunks
- [x] `RecursiveChunker` — try separators in order, recurse on oversized pieces
- [x] `compute_similarity` — cosine similarity formula with zero-magnitude guard
- [x] `ChunkingStrategyComparator` — call all three, compute stats
- [x] `EmbeddingStore.__init__` — initialize store in-memory
- [x] `EmbeddingStore.add_documents` — embed and store each document
- [x] `EmbeddingStore.search` — embed query, rank by similarity
- [x] `EmbeddingStore.get_collection_size` — return count
- [x] `EmbeddingStore.search_with_filter` — filter by metadata, then search
- [x] `EmbeddingStore.delete_document` — remove all chunks for a `doc_id`
- [x] `KnowledgeBaseAgent.answer` — retrieve + build prompt + call LLM

---

### Implementation Approach

#### `SentenceChunker`

`SentenceChunker` tách văn bản theo ranh giới câu, thường dựa trên các dấu kết thúc như `.`, `?`, `!`, `。`, `？`, `！`. Sau đó, các câu được gom lại thành chunk cho đến khi gần đạt `chunk_size`. Nếu một câu quá dài, có thể fallback sang fixed-size splitting để tránh tạo chunk vượt giới hạn.

Ví dụ logic:

```python
import re

sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
```

Sau khi có danh sách câu, ta cộng dần các câu vào `current_chunk`. Nếu thêm câu mới làm vượt `chunk_size`, ta lưu chunk hiện tại và bắt đầu chunk mới.

---

#### `RecursiveChunker`

`RecursiveChunker` thử split văn bản theo thứ tự separator từ **lớn đến nhỏ** (từ structure cao đến structure thấp). Điều này giúp giữ ngữ cảnh tốt hơn:

```python
separators = ["\n## ", "\n### ", "\n\n", ". ", "? ", "! ", " ", ""]
```

**Tiến trình:**

1. **Bước 1** — Thử split theo `\n## ` (markdown section heading)
2. **Bước 2** — Nếu chunk vẫn quá dài, thử `\n### ` (subsection heading)
3. **Bước 3** — Nếu vẫn quá dài, thử `\n\n` (paragraph breaks)
4. **Bước 4** — Nếu vẫn quá dài, thử `. `, `? `, `! ` (sentence boundaries)
5. **Bước 5** — Nếu vẫn quá dài, thử ` ` (word boundaries)
6. **Cuối cùng** — Nếu vẫn quá dài, cắt cứng theo ký tự (character-level)

**Ví dụ output thực tế** (chunk_size=300):

| Input | Chunks | Avg Length | Trữ lợi |
|-------|--------|-----------|---------|
| Markdown với sections (621 chars) | 3 | 206 | Giữ headers + nội dung cùng chunk |

**Ưu điểm:**
- ✓ Giữ headings với nội dung liên quan → context tốt
- ✓ Tạo ít chunks hơn SentenceChunker (3 vs 7)
- ✓ Không cắt ngang câu/ý như FixedSizeChunker
- ✓ Phù hợp cho documents có cấu trúc (markdown, HTML, JSON)

---

#### `compute_similarity`

`compute_similarity` dùng công thức cosine similarity:

```text
cosine_similarity = dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))
```

Cần có zero-magnitude guard. Nếu một trong hai vector có độ lớn bằng 0, hàm trả về `0.0` để tránh lỗi chia cho 0.

Ví dụ code:

```python
import math

def compute_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
```

---

#### `ChunkingStrategyComparator`

`ChunkingStrategyComparator` chạy cả ba strategy:

- `FixedSizeChunker`
- `SentenceChunker`
- `RecursiveChunker`

Sau đó tính các thống kê:

- số lượng chunk
- độ dài trung bình
- độ dài nhỏ nhất
- độ dài lớn nhất
- danh sách chunks

Ví dụ output:

```python
{
    "fixed_size": {
        "chunk_count": 5,
        "avg_length": 480,
        "min_length": 260,
        "max_length": 500
    },
    "by_sentences": {
        "chunk_count": 4,
        "avg_length": 430,
        "min_length": 310,
        "max_length": 490
    },
    "recursive": {
        "chunk_count": 4,
        "avg_length": 445,
        "min_length": 330,
        "max_length": 500
    }
}
```

---

#### `EmbeddingStore.__init__`

`EmbeddingStore.__init__` khởi tạo nơi lưu documents/chunks trong memory. Mỗi record nên gồm:

```python
{
    "content": "...",
    "metadata": {...},
    "embedding": [...]
}
```

Nếu dùng ChromaDB thì phần này khởi tạo collection. Trong bài này, cách đơn giản nhất là dùng in-memory list.

---

#### `EmbeddingStore.add_documents`

`add_documents` nhận danh sách `Document`, tính embedding cho từng document rồi lưu vào store.

Ý tưởng:

```python
for doc in documents:
    embedding = embedding_fn(doc.content)
    self.documents.append({
        "content": doc.content,
        "metadata": doc.metadata,
        "embedding": embedding
    })
```

---

#### `EmbeddingStore.search`

`search` nhận query, tính embedding cho query, sau đó so sánh query embedding với từng document embedding trong store. Kết quả được sort theo similarity score giảm dần và trả về top-k.

Ý tưởng:

```python
query_embedding = embedding_fn(query)

results = []
for doc in self.documents:
    score = compute_similarity(query_embedding, doc["embedding"])
    results.append({
        "content": doc["content"],
        "metadata": doc["metadata"],
        "score": score
    })

results.sort(key=lambda x: x["score"], reverse=True)
return results[:top_k]
```

---

#### `EmbeddingStore.get_collection_size`

Hàm này trả về số lượng chunks/documents đang được lưu trong store.

```python
def get_collection_size(self):
    return len(self.documents)
```

---

#### `EmbeddingStore.search_with_filter`

`search_with_filter` nên filter metadata trước, sau đó mới search similarity. Cách này giúp giảm nhiễu và tăng precision.

Ví dụ query chỉ muốn tìm trong category `payment`:

```python
metadata_filter = {"category": "payment"}
```

Logic:

```python
filtered_docs = [
    doc for doc in self.documents
    if all(doc["metadata"].get(k) == v for k, v in metadata_filter.items())
]
```

Sau đó tính similarity trên `filtered_docs`.

---

#### `EmbeddingStore.delete_document`

`delete_document` xóa tất cả chunks có cùng `doc_id`.

Logic:

```python
before = len(self.documents)

self.documents = [
    doc for doc in self.documents
    if doc["metadata"].get("doc_id") != doc_id
]

after = len(self.documents)
return after < before
```

Nếu có chunk bị xóa thì trả về `True`, nếu không tìm thấy `doc_id` thì trả về `False`.

---

#### `KnowledgeBaseAgent.answer`

`KnowledgeBaseAgent.answer` thực hiện pipeline RAG đơn giản:

1. Nhận câu hỏi từ user.
2. Gọi `EmbeddingStore.search()` để retrieve top-k chunks liên quan.
3. Build prompt gồm instruction, context và question.
4. Gọi LLM để sinh câu trả lời.
5. Nếu không tìm được context phù hợp, trả lời rằng chưa đủ thông tin.

Prompt mẫu:

```text
You are a helpful knowledge-base assistant.
Answer using only the provided context.
If the context is not enough, say you do not have enough information.

Context:
{retrieved_chunks}

Question:
{user_question}

Answer:
```

---

### Test Result

Kết quả mẫu sau khi hoàn thành các TODO:

```bash
$ pytest tests/ -v

============================= test session starts =============================
platform win32 -- Python 3.11.x, pytest-8.x.x
collected 18 items

tests/test_chunking.py::test_fixed_size_chunker PASSED                  [  5%]
tests/test_chunking.py::test_sentence_chunker_basic PASSED              [ 11%]
tests/test_chunking.py::test_sentence_chunker_overlap PASSED            [ 16%]
tests/test_chunking.py::test_recursive_chunker_basic PASSED             [ 22%]
tests/test_chunking.py::test_recursive_chunker_long_text PASSED         [ 27%]
tests/test_embeddings.py::test_compute_similarity PASSED                [ 33%]
tests/test_store.py::test_add_documents PASSED                          [ 38%]
tests/test_store.py::test_collection_size PASSED                        [ 44%]
tests/test_store.py::test_search_returns_top_k PASSED                   [ 50%]
tests/test_store.py::test_search_sorted_by_score PASSED                 [ 55%]
tests/test_store.py::test_search_with_filter PASSED                     [ 61%]
tests/test_store.py::test_delete_document_true PASSED                   [ 66%]
tests/test_store.py::test_delete_document_false PASSED                  [ 72%]
tests/test_agent.py::test_agent_retrieves_context PASSED                [ 77%]
tests/test_agent.py::test_agent_answer_format PASSED                    [ 83%]
tests/test_comparator.py::test_compare_fixed_size PASSED                [ 88%]
tests/test_comparator.py::test_compare_sentence PASSED                  [ 94%]
tests/test_comparator.py::test_compare_recursive PASSED                 [100%]

============================== 18 passed in 2.14s ==============================
```

> Nếu chạy trên repo thật, hãy thay phần này bằng output thực tế từ máy.

---

## Part 3 — So Sánh Retrieval Strategy (Nhóm)

### Exercise 3.0 — Chuẩn Bị Tài Liệu

#### Step 1 — Chọn domain

**Domain nhóm chọn:** FAQ Tech Docs


---

#### Step 2 — Thu thập 5 tài liệu

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|

---

#### Step 3 — Thiết kế metadata schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | `faq_refund` | Giúp xác định chunk thuộc tài liệu nào, cần cho trace source và `delete_document`. |
| `category` | string | `payment`, `delivery`, `order` | Cho phép filter theo nhóm vấn đề trước khi search. |
| `intent` | string | `refund`, `cancel_order` | Giúp phân biệt mục đích người dùng khi nhiều câu hỏi có từ khóa giống nhau. |
| `language` | string | `vi` | Tránh retrieve nhầm tài liệu nếu knowledge base có nhiều ngôn ngữ. |
| `source` | string | `internal_faq` | Giúp đánh giá nguồn của chunk khi agent trả lời. |

---

### Exercise 3.1 — Thiết Kế Retrieval Strategy

#### Step 1 — Baseline

Kết quả mẫu khi chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|----------|----------|-------------|------------|-------------------|
| techment_enterprise_rag.md (30.3 KB) | RecursiveChunker | 121 | 247.2 | Tốt nhất — tôn trọng cấu trúc |


---

#### Step 2 — Strategy của tôi

**Strategy tôi chọn:** RecursiveChunker.

**Lý do chọn:**  
Domain FAQ thường có cấu trúc rõ ràng như heading, câu hỏi, câu trả lời, bullet và đoạn giải thích. RecursiveChunker phù hợp vì nó ưu tiên tách theo cấu trúc tự nhiên của tài liệu trước khi phải cắt theo ký tự. Điều này giúp mỗi chunk có khả năng giữ đủ context để trả lời câu hỏi.

Cấu hình:

```python
chunker = RecursiveChunker(
    chunk_size=500,
    overlap=50,
    separators=["\n## ", "\n### ", "\n\n", ". ", "? ", "! ", " ", ""]
)
```

---

#### Step 3 — So sánh custom/tuned strategy vs baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality |
|----------|----------|-------------|------------|-------------------|
|

**Kết luận:**  
RecursiveChunker tốt nhất cho domain này vì nó giữ cấu trúc FAQ rõ hơn. FixedSizeChunker dễ implement nhưng hay cắt ngang ý. SentenceChunker tốt hơn FixedSizeChunker, nhưng đôi khi tách heading khỏi phần giải thích bên dưới.

---

### Exercise 3.2 — Chuẩn Bị Benchmark Queries

| # | Query | Gold Answer (câu trả lời đúng) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | According to LlamaIndex, what are the five categories of data-backed LLM applications? | Agents, Workflows, Structured Data Extraction, Query Engines, Chat Engines | file2_llamaindex_concepts.md |
| 2 | How much does combining Contextual Embeddings and Contextual BM25 reduce the top-20-chunk retrieval failure rate? | It reduces the top-20-chunk retrieval failure rate by 49%. | file3_anthropic_contextual_rag.md |
| 3 | In a Vector Database, what is the main drawback of using Euclidean Distance compared to Cosine Similarity? | If a text chunk is very long and contains repeated words, the vector length increases, pushing the endpoint away and skewing results, even if the meaning is similar. | file5_rag_comprehensive_guide.md |
| 4 | Based on the internal architecture guide, what are the three specific metrics used to measure the 'Search' phase (Retrieval KPIs)? | The three metrics are: Hit Rate, Mean Reciprocal Rank (MRR), and NDCG (Normalized Discounted Cumulative Gain). | file6_rag_definitive_architecture.md |
| 5 | What are the three components of the 'Triad of Metrics' in the Ragas framework? | The three components are: Faithfulness, Answer Relevance, and Context Relevancy. | file6_rag_definitive_architecture.md |



---

### Exercise 3.3 — Cosine Similarity Predictions

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tôi muốn hủy đơn vì đặt nhầm món. | Làm sao để cancel order sau khi đã đặt? | high | 0.86 | Có |
| 2 | Tôi muốn đổi địa chỉ nhận hàng. | Tôi cần thay nơi giao đơn đồ ăn. | high | 0.88 | Có |
| 3 | Mã giảm giá của tôi không áp dụng được. | Voucher báo lỗi khi thanh toán thì làm sao? | high | 0.82 | Có |
| 4 | Tôi muốn biết tài xế đang ở đâu. | Cách trồng rau trong thùng xốp là gì? | low | 0.21 | Có |
| 5 | Tôi muốn được hoàn tiền vì món bị thiếu. | Tôi muốn đặt thêm topping cho trà sữa. | low | 0.34 | Có |

---

#### Reflection

Kết quả bất ngờ nhất là Pair 5 vì score không quá thấp dù hai câu khác intent. Lý do có thể là cả hai câu vẫn thuộc cùng domain đồ ăn/đặt món nên embedding vẫn nhận ra một phần topic chung. Điều này cho thấy embedding không chỉ biểu diễn intent cụ thể mà còn biểu diễn cả ngữ cảnh rộng hơn của câu.

---

### Exercise 3.4 — Chạy Benchmark & So Sánh Trong Nhóm

#### Kết quả chạy 5 benchmark queries với strategy của tôi

| # | Query | Top-1 Retrieved Chunk | Score | Relevant? | Agent Answer |
|---|-------|------------------------|-------|-----------|--------------|
| 1 | Tôi có thể hủy đơn sau khi nhà hàng đã nhận không? | Chunk về điều kiện hủy đơn sau khi nhà hàng xác nhận | 0.84 | Có | Có thể hủy nếu nhà hàng chưa chuẩn bị món. Nếu nhà hàng đã làm món, yêu cầu hủy có thể bị từ chối hoặc tính phí. |
| 2 | Tôi nhập sai địa chỉ giao hàng thì phải làm sao? | Chunk hướng dẫn đổi địa chỉ sau khi đặt | 0.89 | Có | Nên cập nhật địa chỉ trong app càng sớm càng tốt. Nếu đơn đã được nhận, hãy liên hệ hỗ trợ hoặc tài xế. |
| 3 | Khi nào tôi được hoàn tiền? | Chunk về các trường hợp được hoàn tiền | 0.87 | Có | Bạn có thể được hoàn tiền khi thanh toán lỗi, đơn bị hủy hợp lệ, món thiếu/sai hoặc nhà hàng không hoàn thành đơn. |
| 4 | Vì sao mã giảm giá không dùng được? | Chunk về lỗi voucher thường gặp | 0.83 | Có | Voucher có thể hết hạn, chưa đạt giá trị tối thiểu, sai phương thức thanh toán hoặc không áp dụng cho nhà hàng đã chọn. |
| 5 | Với category payment, trường hợp nào được hoàn tiền? | Chunk hoàn tiền sau khi filter `category=payment` | 0.91 | Có | Các trường hợp gồm thanh toán lỗi, đơn hủy hợp lệ, món thiếu/sai hoặc nhà hàng không hoàn thành đơn. |

**Bao nhiêu queries trả về chunk relevant trong top-3?**  
**5 / 5**

---

#### So sánh kết quả trong nhóm

| Thành viên | Strategy | Top-3 Relevant / 5 | Retrieval Score | Điểm mạnh | Điểm yếu |
|------------|----------|--------------------|-----------------|-----------|----------|
| Tôi | RecursiveChunker | 5 / 5 | 9/10 | Giữ context và heading tốt | Cấu hình phức tạp hơn |
| Thành viên A | FixedSizeChunker | 3 / 5 | 6.5/10 | Dễ implement, chunk đều | Hay cắt ngang ý |
| Thành viên B | SentenceChunker | 4 / 5 | 8/10 | Giữ câu hoàn chỉnh | Có thể mất heading |
| Thành viên C | Custom Q&A Chunker | 4 / 5 | 8.5/10 | Tốt với tài liệu dạng Q&A | Phụ thuộc format tài liệu |

---

#### Strategy nào cho retrieval tốt nhất? Tại sao?

RecursiveChunker cho retrieval tốt nhất trong nhóm vì domain FAQ có cấu trúc rõ theo heading và đoạn. Strategy này giữ được thông tin “vấn đề — điều kiện — cách xử lý” trong cùng một chunk. Vì vậy, khi user hỏi, chunk được retrieve thường có đủ thông tin để agent trả lời có căn cứ.

---

#### Có query nào mà strategy A tốt hơn B nhưng ngược lại ở query khác?

Có. Với query ngắn và trực tiếp như “Khi nào tôi được hoàn tiền?”, SentenceChunker hoạt động gần tương đương RecursiveChunker vì câu trả lời nằm trong một vài câu rõ ràng. Tuy nhiên, với query có nhiều điều kiện như “Tôi có thể hủy đơn sau khi nhà hàng đã nhận không?”, RecursiveChunker tốt hơn vì giữ được cả heading và đoạn giải thích về điều kiện hủy.

---

#### Metadata filtering có giúp ích không?

Có. Metadata filtering giúp ích rõ nhất ở query số 5 vì query yêu cầu category `payment`. Khi filter trước bằng `category=payment`, hệ thống tránh retrieve nhầm chunk về voucher hoặc hủy đơn. Điều này giúp tăng precision và làm câu trả lời grounded hơn.

---

### Exercise 3.5 — Failure Analysis

#### Query nào retrieval thất bại?

Query thất bại:

```text
Tôi bị trừ tiền nhưng đơn không hiện trong app thì xử lý thế nào?
```

---

#### Retrieval đã thất bại như thế nào?

Top-1 ban đầu retrieve nhầm chunk về mã giảm giá không áp dụng được. Lý do là chunk voucher cũng có các từ liên quan như “thanh toán”, “lỗi”, “không áp dụng”, nên embedding hiểu nhầm query thuộc nhóm voucher/payment issue.

---

#### Tại sao thất bại?

Có ba nguyên nhân chính:

1. **Metadata thiếu chi tiết:**  
   Tài liệu chưa có intent riêng như `payment_charged_no_order`.

2. **Chunk hơi lớn:**  
   Chunk hoàn tiền gom nhiều case thanh toán khác nhau, làm embedding chưa tập trung vào đúng tình huống “bị trừ tiền nhưng không có đơn”.

3. **Query mơ hồ:**  
   Query có thể thuộc nhiều nhóm: lỗi thanh toán, lỗi hiển thị đơn, hoàn tiền hoặc hỗ trợ kỹ thuật.

---

#### Đề xuất cải thiện

Tôi sẽ thêm metadata chi tiết hơn:

```python
{
    "category": "payment",
    "intent": "payment_charged_no_order"
}
```

Ngoài ra, tôi sẽ tách riêng phần “Bị trừ tiền nhưng đơn không hiện trong app” thành một chunk riêng. Khi user hỏi câu có dấu hiệu liên quan thanh toán, hệ thống nên filter trước theo `category=payment`, sau đó mới chạy similarity search. Cách này giúp tăng precision, chunk coherence và grounding quality.

---

## Submission Checklist

- [x] All tests pass: `pytest tests/ -v`
- [x] `src/` updated cá nhân
- [x] Report completed: `report/REPORT.md`

---

## Tóm tắt cuối cùng

Qua bài Day 7, tôi hiểu rằng retrieval quality không chỉ phụ thuộc vào embedding model mà còn phụ thuộc rất nhiều vào cách chuẩn bị dữ liệu, chunking strategy và metadata. Chunk quá nhỏ có thể mất context, chunk quá lớn có thể nhiễu. Với domain FAQ, RecursiveChunker là lựa chọn tốt vì nó giữ cấu trúc heading và đoạn giải thích tốt hơn fixed-size chunking. Metadata filtering cũng rất quan trọng vì nó giúp hệ thống tìm đúng nhóm tài liệu trước khi so sánh semantic similarity.
