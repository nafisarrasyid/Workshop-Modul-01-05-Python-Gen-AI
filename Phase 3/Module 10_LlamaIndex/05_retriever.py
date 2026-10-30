"""Module 10 - LlamaIndex
Section 10.5: Retriever & Response Synthesizer

Demonstrates:
  - Decoupling Retrieval from Synthesis
  - Using index.as_retriever() and VectorIndexRetriever directly
  - Filtering and inspecting retrieved nodes prior to LLM synthesis
  - Response synthesis modes: 'compact', 'tree_summarize', 'refine'
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import configure_llama_index

from llama_index.core import Document, VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

def main():
    print("=" * 60)
    print("Module 10.5: Decoupled Retrieval and Response Synthesis")
    print("=" * 60)

    configure_llama_index()

    docs = [
        Document(
            text="Hugging Face Transformers provides thousands of pretrained models for NLP, vision, and audio.",
            metadata={"topic": "models", "library": "transformers"},
        ),
        Document(
            text="PEFT (Parameter-Efficient Fine-Tuning) enables LoRA adaptation with less than 1% trainable parameters.",
            metadata={"topic": "fine_tuning", "library": "peft"},
        ),
        Document(
            text="Datasets library allows streaming and preprocessing large ML datasets with PyArrow memory mapping.",
            metadata={"topic": "data", "library": "datasets"},
        ),
        Document(
            text="Accelerate simplifies distributed PyTorch training across multiple GPUs and TPUs.",
            metadata={"topic": "distributed", "library": "accelerate"},
        ),
    ]

    index = VectorStoreIndex.from_documents(docs)

    # 1. Custom Vector Index Retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=2,
    )

    query = "How can I fine-tune models efficiently with minimal memory?"
    print(f"Retrieving nodes for query: '{query}'")

    retrieved_nodes = retriever.retrieve(query)
    print(f"\nRetrieved {len(retrieved_nodes)} relevant nodes:")
    for idx, node in enumerate(retrieved_nodes, start=1):
        print(f"  [{idx}] {node.node.text} (Score: {node.score if node.score is not None else 'N/A'})")

    # 2. Response Synthesizer with 'compact' mode
    synthesizer = get_response_synthesizer(response_mode="compact")

    # 3. Assemble Custom RetrieverQueryEngine
    custom_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synthesizer,
    )

    response = custom_engine.query(query)
    print(f"\nSynthesized Response:\n{response}")

    print("\nRetriever and Synthesizer pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
