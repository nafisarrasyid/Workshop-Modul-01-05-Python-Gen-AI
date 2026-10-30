"""Module 10 - LlamaIndex
Section 10.7: Persistent Storage & StorageContext

Demonstrates:
  - Persisting indexes, document stores, and vector stores to disk
  - Reloading a persisted index via StorageContext.from_defaults(persist_dir=...)
  - Verifying queries against the reloaded index without recomputing embeddings
"""
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import configure_llama_index

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)

def main():
    print("=" * 60)
    print("Module 10.7: Index Persistence & StorageContext")
    print("=" * 60)

    configure_llama_index()

    persist_dir = Path(__file__).resolve().parent / "storage_demo"
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    docs = [
        Document(
            text="Antigravity is an advanced agentic AI coding environment designed for complex software development.",
            metadata={"product": "Antigravity", "category": "IDE"},
        ),
        Document(
            text="OmniRoute provides unified API proxying and routing across multiple LLM backends.",
            metadata={"product": "OmniRoute", "category": "Infrastructure"},
        ),
    ]

    print("Step 1: Creating index and persisting to disk...")
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    # Persist all components (docstore, index_store, vector_store)
    index.storage_context.persist(persist_dir=str(persist_dir))
    print(f"Persisted index artifacts to: {persist_dir}")

    for file in persist_dir.glob("*.json"):
        print(f"  Saved file: {file.name} ({file.stat().st_size} bytes)")

    print("\nStep 2: Reloading index from disk...")
    # Load storage context from persisted directory
    reloaded_storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    reloaded_index = load_index_from_storage(reloaded_storage_context)

    print("Index reloaded successfully!")

    # Step 3: Query the reloaded index
    query_engine = reloaded_index.as_query_engine(similarity_top_k=1)
    query = "What is Antigravity?"
    print(f"\nQuerying reloaded index: '{query}'")

    response = query_engine.query(query)
    print(f"Response:\n{response}")

    print("\nPersistence workflow completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
