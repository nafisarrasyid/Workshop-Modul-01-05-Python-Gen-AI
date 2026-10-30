"""Module 10 - LlamaIndex
Section 10.6: SummaryIndex (ListIndex)

Demonstrates:
  - Creating a SummaryIndex without embedding overhead
  - Using tree_summarize and default response modes
  - Comparing SummaryIndex vs VectorStoreIndex for full-text summarization
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import configure_llama_index

from llama_index.core import Document, SummaryIndex

def main():
    print("=" * 60)
    print("Module 10.6: LlamaIndex SummaryIndex")
    print("=" * 60)

    configure_llama_index()

    # Chapters of a technical report
    chapters = [
        Document(
            text="Chapter 1: Agent architectures have evolved from simple ReAct loops into structured StateGraphs and collaborative multi-agent teams.",
            metadata={"chapter": 1},
        ),
        Document(
            text="Chapter 2: Vector databases provide sub-millisecond similarity search across millions of high-dimensional vectors.",
            metadata={"chapter": 2},
        ),
        Document(
            text="Chapter 3: PEFT and quantization techniques like QLoRA democratized LLM customization by reducing VRAM requirements by up to 75%.",
            metadata={"chapter": 3},
        ),
    ]

    print(f"Building SummaryIndex across {len(chapters)} document chapters...")
    summary_index = SummaryIndex.from_documents(chapters)

    # 1. Summary Query Engine with tree_summarize
    query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize"
    )

    query = "Summarize the key takeaways across all chapters of the technical report."
    print(f"\nExecution Query: '{query}'")

    response = query_engine.query(query)
    print(f"\nSummary Response:\n{response}")

    print("\nSummaryIndex workflow completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
