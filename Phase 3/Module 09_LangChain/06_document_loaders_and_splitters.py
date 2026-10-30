# Module 09 - LangChain
# 9.6 Document Loaders and Text Splitters
#
# In RAG applications, raw documents must be:
#   1. Loaded into standardized LangChain Document objects (content + metadata).
#   2. Chunked using text splitters to fit context windows and preserve semantic units.
#
# Key components:
#   - Document (langchain_core.documents)
#   - TextLoader (langchain_community.document_loaders)
#   - RecursiveCharacterTextSplitter (langchain_text_splitters)

from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter


def run_loaders_and_splitters_demo() -> None:
    print("=" * 60, flush=True)
    print("9.6 Document Loaders & Text Splitters Demo", flush=True)
    print("=" * 60, flush=True)

    # ── 1. Creating and Structuring In-Memory Documents ──────────────────────
    print("\n--- 1. In-Memory Documents with Metadata ---", flush=True)
    raw_docs = [
        Document(
            page_content="""LangChain provides a standard interface for interacting with
large language models and related tools. It simplifies the development of
context-aware reasoning applications by combining prompts, models, memory,
and retrieval in composable pipelines. LCEL makes these pipelines
declarative and easy to debug.""",
            metadata={"source": "langchain_docs.txt", "page": 1, "topic": "framework"}
        ),
        Document(
            page_content="""Retrieval-Augmented Generation (RAG) is a pattern that combines
a retriever with a generator. The retriever finds relevant documents from a
knowledge base. Those documents are injected into the generator's context window.
This allows models to answer questions about information they were never trained on.""",
            metadata={"source": "rag_guide.txt", "page": 1, "topic": "rag"}
        ),
    ]

    print(f"Loaded {len(raw_docs)} documents.")
    for i, doc in enumerate(raw_docs, 1):
        print(f"  Doc {i}: source={doc.metadata['source']} | length={len(doc.page_content)} chars")

    # ── 2. RecursiveCharacterTextSplitter (Hierarchical Splitting) ───────────
    print("\n--- 2. RecursiveCharacterTextSplitter (size=200, overlap=40) ---", flush=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=40,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(raw_docs)
    print(f"Original docs: {len(raw_docs)}, Resulting Chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        preview = chunk.page_content.replace("\n", " ")[:60]
        print(f"  Chunk {i}: {len(chunk.page_content)} chars | source: {chunk.metadata['source']} | preview: '{preview}...'")

    # ── 3. File Loader Demo (TextLoader) ─────────────────────────────────────
    print("\n--- 3. File System Ingestion with TextLoader ---", flush=True)
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    sample_file = data_dir / "sample_manual.txt"
    sample_file.write_text(
        "OmniRoute Architecture Manual\n\n"
        "OmniRoute acts as a local proxy multiplexing AI provider calls.\n"
        "It exposes an OpenAI-compatible /v1/chat/completions route.\n"
        "It transparently routes requests to local or remote endpoints.",
        encoding="utf-8"
    )

    loader = TextLoader(str(sample_file), encoding="utf-8")
    loaded_docs = loader.load()
    print(f"Successfully loaded file via TextLoader: {sample_file.name}")
    print(f"Loaded page content: '{loaded_docs[0].page_content.splitlines()[0]}' | metadata={loaded_docs[0].metadata}")

    # ── 4. Chunk Overlap Verification ────────────────────────────────────────
    print("\n--- 4. Chunk Overlap Verification ---", flush=True)
    test_text = "Step 1: Ingest documents. Step 2: Chunk with overlap. Step 3: Embed vectors. Step 4: Index into database."
    custom_splitter = RecursiveCharacterTextSplitter(chunk_size=45, chunk_overlap=15)
    test_chunks = custom_splitter.split_text(test_text)
    for idx, c in enumerate(test_chunks):
        print(f"  Segment {idx}: '{c}'")

    print("\n" + "=" * 60, flush=True)
    print("9.6 Document Loaders & Text Splitters Demo Completed Successfully!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_loaders_and_splitters_demo()
