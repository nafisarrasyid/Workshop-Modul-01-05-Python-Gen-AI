# Phase 3 - GenAI Libraries: Module 09 (LangChain)

Repositori ini berisi implementasi kode lengkap untuk **Phase 3 - Module 09 (LangChain)** berdasarkan panduan panduan materi *Python for Gen AI*:

1. **9.2 Chat Models** -> [`02_chat_models.py`](file:///n:/napis/Gen%20Ai/Phase%203/02_chat_models.py)
2. **9.3 Prompt Templates** -> [`03_prompt_templates.py`](file:///n:/napis/Gen%20Ai/Phase%203/03_prompt_templates.py)
3. **9.4 LCEL - LangChain Expression Language, Branching with RunnablePassthrough, Conditional routing** -> [`04_lcel_and_routing.py`](file:///n:/napis/Gen%20Ai/Phase%203/04_lcel_and_routing.py)
4. **9.5 Output Parsers** -> [`05_output_parsers.py`](file:///n:/napis/Gen%20Ai/Phase%203/05_output_parsers.py)
5. **9.6 Document Loaders and Text Splitters** -> [`06_document_loaders_and_splitters.py`](file:///n:/napis/Gen%20Ai/Phase%203/06_document_loaders_and_splitters.py)
6. **9.7 Conversational Memory** -> [`07_conversational_memory.py`](file:///n:/napis/Gen%20Ai/Phase%203/07_conversational_memory.py)
7. **9.8 Building a Full RAG Chain with LangChain** -> [`08_full_rag_chain.py`](file:///n:/napis/Gen%20Ai/Phase%203/08_full_rag_chain.py)
8. **9.9 Module 09 Exercises** -> [`09_exercises.py`](file:///n:/napis/Gen%20Ai/Phase%203/09_exercises.py)

---

## 🛠️ Persiapan dan Dependensi yang Telah Diinstall

Library yang dibutuhkan telah dipasang dan kompatibel dengan Python 3.9:
- `langchain` & `langchain-core` (LCEL, Prompts, Runnables, Output Parsers)
- `langchain-community` (Vector stores & Document Loaders)
- `langchain-openai` & `langchain-anthropic` (Model Wrappers)
- `langchain-text-splitters` (RecursiveCharacterTextSplitter)
- `pydantic` (Validasi skema JSON & structured output)
- `python-dotenv` (Pemuatan environment variables)
- `tiktoken` (Tokenisasi)

---

## 🚀 Cara Menjalankan Kode

Kamu bisa menjalankan masing-masing skrip secara independen:

```bash
cd "n:\napis\Gen Ai\Phase 3"

# 9.2 Chat Models (Invoke, Stream, Batch)
python 02_chat_models.py

# 9.3 Prompt Templates & MessagesPlaceholder
python 03_prompt_templates.py

# 9.4 LCEL, RunnableParallel Branching & Conditional Routing
python 04_lcel_and_routing.py

# 9.5 Output Parsers (Pydantic with_structured_output, JsonOutputParser, StrOutputParser)
python 05_output_parsers.py

# 9.6 Document Loaders (TextLoader) & RecursiveCharacterTextSplitter
python 06_document_loaders_and_splitters.py

# 9.7 Conversational Memory (RunnableWithMessageHistory)
python 07_conversational_memory.py

# 9.8 Full RAG Chain (Ingest -> Split -> Embed -> Retrieve -> Grounded Generation)
python 08_full_rag_chain.py

# 9.9 Module 09 Exercises (4 Latihan Lengkap)
python 09_exercises.py

# Atau jalankan seluruh test suite sekaligus:
python run_all.py
```

---

## 📌 Penjelasan Detail Solusi Latihan 9.9 ([`09_exercises.py`](file:///n:/napis/Gen%20Ai/Phase%203/09_exercises.py))

1. **Latihan 1 (Fact Extraction -> Quiz Generator Chain)**:
   - Chain LCEL dua tahap: Ekstraksi 3 fakta penting dari teks dokumen mentah (Tahap 1), lalu menghasilkan pertanyaan kuis untuk tiap fakta (Tahap 2). Mengembalikan hasil akhir berupa Python `list[str]`.
2. **Latihan 2 (Manual MultiQueryRetriever)**:
   - Mengembangkan class kustom yang melakukan query expansion menjadi 3 variasi parafrasa query, melakukan retrieval ke VectorStore untuk setiap variasi query, lalu melakukan deduplikasi berbasis hash SHA-256 konten dokumen sebelum mengembalikan gabungan (union) dokumen unik.
3. **Latihan 3 (Map-Reduce Long Document Summarization)**:
   - Memecah dokumen berukuran panjang menggunakan `RecursiveCharacterTextSplitter`. Tahap *Map* membuat ringkasan per chunk, lalu digabungkan dan diproses oleh tahap *Reduce* untuk menghasilkan ringkasan eksekutif akhir yang koheren.
4. **Latihan 4 (RAG Quality & Groundedness Scoring Filter 0-10)**:
   - Menambahkan auditor/evaluator LLM pada RAG chain dari section 9.8. Setelah jawaban dihasilkan, evaluator memeriksa apakah jawaban tersebut didukung sepenuhnya oleh konteks yang diretrieve dan memberikan skor 0–10 beserta alasannya.
