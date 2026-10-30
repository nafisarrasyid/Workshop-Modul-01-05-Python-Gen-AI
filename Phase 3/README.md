# Phase 3: GenAI Libraries - Complete Suite

Comprehensive implementation and test suite for Phase 3 covering **Modules 09 through 14** of the *Python for GenAI* curriculum.

---

## Folder Organization

```
n:/napis/Gen Ai/
├── Phase 3/
│   ├── Module 09_LangChain/
│   ├── Module 10_LlamaIndex/
│   ├── Module 11_VectorDatabases/
│   ├── Module 12_HuggingFaceTransformers/
│   ├── Module 13_LangGraph/
│   ├── Module 14_CrewAI/
│   ├── utils.py                     # Shared Model Resolvers & Patches
│   ├── run_all_phase3.py            # Master Test Runner (35/35 passing)
│   ├── requirements.txt
│   └── .env                         # API Keys (OpenAI, OmniRoute, etc.)
│
├── Module 09_LangChain/             # Mirrored at Root for convenience
├── Module 10_LlamaIndex/
├── Module 11_VectorDatabases/
├── Module 12_HuggingFaceTransformers/
├── Module 13_LangGraph/
└── Module 14_CrewAI/
```

---

## Modules Overview

### 1. Module 09 - LangChain
- **02_chat_models.py**: Chat models, streaming, temperature, structured invocation.
- **03_prompt_templates.py**: ChatPromptTemplate, System/Human messages, parameter formatting.
- **04_lcel_and_routing.py**: LangChain Expression Language (LCEL), `RunnableParallel`, `RunnableBranch`, conditional routing.
- **05_output_parsers.py**: StrOutputParser, JsonOutputParser, PydanticOutputParser.
- **06_document_loaders_and_splitters.py**: Text loaders, RecursiveCharacterTextSplitter.
- **07_conversational_memory.py**: RunnableWithMessageHistory, InMemoryChatMessageHistory.
- **08_full_rag_chain.py**: End-to-end RAG with vector search and context-grounded synthesis.
- **09_exercises.py**: All 4 Module 09 exercises.

### 2. Module 10 - LlamaIndex
- **02_setup_and_settings.py**: Global `Settings` (LLM, Embedding, chunk_size, chunk_overlap).
- **03_documents_and_nodes.py**: Documents, metadata, SentenceSplitter, TextNode relationships.
- **04_vector_store_index.py**: VectorStoreIndex construction and query engine.
- **05_retriever.py**: Decoupled VectorIndexRetriever and ResponseSynthesizer.
- **06_summary_index.py**: SummaryIndex (ListIndex) for full-document tree summarization.
- **07_persistent_storage.py**: StorageContext disk persistence and index reloading.
- **08_custom_query_pipeline.py**: Custom QA prompt templates and response formatting.
- **09_exercises.py**: All 4 Module 10 exercises.

### 3. Module 11 - Vector Databases
- **02_chromadb.py**: ChromaDB PersistentClient, metadata filtering ($and, $eq, $in), cosine distance.
- **03_pinecone.py**: Pinecone v7 SDK, ServerlessSpec, isolated namespaces, metadata queries.
- **04_weaviate.py**: Weaviate v4 Python client, Collection schemas, Hybrid Search (dense + BM25).
- **05_exercises.py**: Benchmark, MultiCollectionSearch, TenantVectorStore, Migration Tool.

### 4. Module 12 - Hugging Face Transformers
- **02_pipeline_api.py**: Sentiment analysis, Zero-shot classification, Feature extraction.
- **03_tokenizers.py**: AutoTokenizer, Token IDs, attention masks, padding, truncation.
- **04_automodel.py**: AutoModelForCausalLM text generation, greedy vs. temperature vs. top-p sampling.
- **05_peft_lora.py**: Low-Rank Adaptation (LoRA), LoraConfig, parameter efficiency (< 1% trainable).
- **06_exercises.py**: Multi-task pipelines, BPE vs WordPiece comparison, sampling diversity, LoRA budget math.

### 5. Module 13 - LangGraph & Agents
- **02_core_concepts.py**: StateGraph, TypedDict state, nodes, edges, conditional routing.
- **03_react_agent.py**: ReAct reasoning loop with executable `@tool` calls and observations.
- **04_human_in_the_loop.py**: MemorySaver checkpointing, `interrupt_before` approval breakpoints.
- **05_exercises.py**: Research Agent, Max step safety limit, Plan-and-Execute, Multi-turn thread memory.

### 6. Module 14 - CrewAI
- **02_agents.py**: Roles, goals, backstories, delegation permissions.
- **03_tools.py**: Custom BaseTool subclassing and `@tool` functional tools.
- **04_tasks.py**: Task definitions and `context=[...]` dependency chaining.
- **05_crew_sequential.py**: End-to-end sequential multi-agent execution pipeline.
- **06_hierarchical_manager.py**: Hierarchical manager-led coordination and delegation.
- **07_exercises.py**: Content pipeline, Code review crew, Search tool integration, Process comparison.

---

## Running Validations

To run all 35 scripts across Phase 3:
```bash
python run_all_phase3.py
```
**Verification Result**: `35 PASSED, 0 FAILED out of 35 scripts` (100% pass rate).
