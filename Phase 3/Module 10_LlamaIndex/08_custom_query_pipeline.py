"""Module 10 - LlamaIndex
Section 10.8: Custom Prompts & Query Pipelines

Demonstrates:
  - Customizing QA and Refine prompt templates in LlamaIndex
  - Injecting domain-specific instructions (e.g. JSON format, citation guidelines)
  - Inspecting default and updated prompts
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import configure_llama_index

from llama_index.core import Document, VectorStoreIndex, PromptTemplate

def main():
    print("=" * 60)
    print("Module 10.8: Custom Prompt Templates & Query Customization")
    print("=" * 60)

    configure_llama_index()

    docs = [
        Document(
            text="LangGraph introduces cyclical graph execution, allowing agents to iteratively refine code and run validations.",
            metadata={"framework": "LangGraph"},
        ),
        Document(
            text="CrewAI orchestrates teams of agents through role playing, backstories, and task delegation.",
            metadata={"framework": "CrewAI"},
        ),
    ]

    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine(similarity_top_k=2)

    # 1. Define custom QA prompt template
    custom_qa_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, "
        "answer the query in a concise, bulleted format.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    custom_qa_prompt = PromptTemplate(custom_qa_str)

    # 2. Update query engine prompt
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": custom_qa_prompt}
    )

    print("Updated Text QA Template:")
    print(custom_qa_str)

    # 3. Execute query with updated prompt
    query = "How do LangGraph and CrewAI approach multi-agent workflows differently?"
    print(f"\nExecuting Query: '{query}'")

    response = query_engine.query(query)
    print(f"\nResponse:\n{response}")

    print("\nCustom prompt pipeline completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
