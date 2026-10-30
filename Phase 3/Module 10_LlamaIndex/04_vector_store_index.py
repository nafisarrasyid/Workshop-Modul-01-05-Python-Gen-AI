"""Module 10 - LlamaIndex
Section 10.4: VectorStoreIndex

Demonstrates:
  - Building a VectorStoreIndex from Document collections
  - Creating a QueryEngine with custom similarity thresholds
  - Executing semantic queries and inspecting the response source nodes
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import configure_llama_index

from llama_index.core import Document, VectorStoreIndex

def main():
    print("=" * 60)
    print("Module 10.4: LlamaIndex VectorStoreIndex")
    print("=" * 60)

    configure_llama_index()

    # 1. Create a corpus of technical knowledge documents
    docs = [
        Document(
            text="LangChain is focused on chains, agents, and modular LLM pipelines with LCEL.",
            metadata={"framework": "LangChain", "focus": "orchestration"},
        ),
        Document(
            text="LlamaIndex specializes in data indexing, retrieval engines, and RAG pipelines.",
            metadata={"framework": "LlamaIndex", "focus": "data_indexing"},
        ),
        Document(
            text="Vector databases like Chroma, Pinecone, and Weaviate store high-dimensional embeddings for fast similarity search.",
            metadata={"framework": "VectorDB", "focus": "storage"},
        ),
        Document(
            text="LangGraph enables cyclic graphs, persistent multi-agent workflows, and human-in-the-loop control.",
            metadata={"framework": "LangGraph", "focus": "stateful_agents"},
        ),
    ]

    print(f"Indexing {len(docs)} knowledge documents into VectorStoreIndex...")
    index = VectorStoreIndex.from_documents(docs)

    # 2. Build Query Engine with top_k = 2
    query_engine = index.as_query_engine(similarity_top_k=2)

    question = "What framework is best for data indexing and RAG retrieval?"
    print(f"\nUser Query: '{question}'")

    response = query_engine.query(question)
    print(f"\nQuery Engine Response:\n{response}")

    # 3. Inspect retrieved Source Nodes
    print("\nSource Nodes Retrieved:")
    for idx, node_with_score in enumerate(response.source_nodes, start=1):
        score_info = f" (Score: {node_with_score.score:.4f})" if node_with_score.score else ""
        print(f"  [Source {idx}]{score_info}")
        print(f"    Text: {node_with_score.node.text}")
        print(f"    Metadata: {node_with_score.node.metadata}")

    print("\nVectorStoreIndex workflow completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
