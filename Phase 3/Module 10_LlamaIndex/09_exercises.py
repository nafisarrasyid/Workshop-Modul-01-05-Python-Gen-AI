"""Module 10 - LlamaIndex
Section 10.9: Module 10 Exercises

Implements all 4 exercises from the curriculum:
  - Exercise 1: Multi-Document Query Engine with Metadata Filtering
  - Exercise 2: SentenceSplitter & Custom Node Parser Analysis
  - Exercise 3: VectorStoreIndex vs. SummaryIndex Comparison
  - Exercise 4: Incremental Index Updates with Persistent Storage
"""
import sys
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import configure_llama_index

from llama_index.core import (
    Document,
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.node_parser import SentenceSplitter

def exercise_1_metadata_filtered_query():
    print("\n--- Exercise 1: Metadata Filtered Query Engine ---")
    docs = [
        Document(
            text="Cloud Run is a fully managed serverless execution environment for containerized applications.",
            metadata={"domain": "compute", "provider": "GCP"},
        ),
        Document(
            text="AWS Lambda runs serverless event-driven code without provisioning servers.",
            metadata={"domain": "compute", "provider": "AWS"},
        ),
        Document(
            text="BigQuery is a serverless, highly scalable enterprise data warehouse for analytics.",
            metadata={"domain": "database", "provider": "GCP"},
        ),
        Document(
            text="Amazon DynamoDB is a serverless NoSQL key-value database.",
            metadata={"domain": "database", "provider": "AWS"},
        ),
    ]

    index = VectorStoreIndex.from_documents(docs)

    # Filter for GCP only
    gcp_filters = MetadataFilters(
        filters=[ExactMatchFilter(key="provider", value="GCP")]
    )
    engine = index.as_query_engine(filters=gcp_filters, similarity_top_k=2)

    query = "What serverless solutions are available?"
    response = engine.query(query)
    print(f"Query: '{query}' (Filtered to provider == 'GCP')")
    print(f"Response:\n{response}")
    print("Retrieved Sources:")
    for n in response.source_nodes:
        print(f"  - Provider: {n.node.metadata.get('provider')}, Domain: {n.node.metadata.get('domain')}")
        assert n.node.metadata.get("provider") == "GCP", "Filter violation detected!"
    print("Exercise 1 passed successfully!")


def exercise_2_node_parser_analysis():
    print("\n--- Exercise 2: SentenceSplitter & Node Parsing Analysis ---")
    long_doc = Document(
        text=(
            "Retrieval-Augmented Generation (RAG) grounds language models on authoritative external documents. "
            "Chunk size directly impacts both retrieval precision and synthesis quality. "
            "Smaller chunks preserve fine-grained semantic focus but may lose broad conversational context. "
            "Larger chunks provide richer context but risk diluting the specific signal needed for precision answering. "
            "Sentence overlap ensures sentences spanning chunk boundaries are not truncated abruptly."
        ),
        metadata={"title": "RAG Chunking Fundamentals"},
    )

    configs = [(30, 5), (60, 10)]
    for chunk_size, chunk_overlap in configs:
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents([long_doc])
        print(f"\nConfiguration [chunk_size={chunk_size}, overlap={chunk_overlap}]:")
        print(f"  Generated {len(nodes)} nodes.")
        for i, n in enumerate(nodes[:2], 1):
            print(f"    Node {i} snippet: '{n.text[:60]}...' (Length: {len(n.text)} chars)")
    print("Exercise 2 passed successfully!")


def exercise_3_vector_vs_summary_comparison():
    print("\n--- Exercise 3: VectorStoreIndex vs. SummaryIndex Comparison ---")
    articles = [
        Document(text="Part 1: The foundation of transformer architecture is multi-head self-attention.", metadata={"part": 1}),
        Document(text="Part 2: Positional encodings provide sequence order information to parallel attention heads.", metadata={"part": 2}),
        Document(text="Part 3: Feed-forward networks apply non-linear projections after each attention layer.", metadata={"part": 3}),
    ]

    vector_idx = VectorStoreIndex.from_documents(articles)
    summary_idx = SummaryIndex.from_documents(articles)

    vec_engine = vector_idx.as_query_engine(similarity_top_k=1)
    sum_engine = summary_idx.as_query_engine(response_mode="tree_summarize")

    query_specific = "What role does positional encoding play?"
    query_broad = "Summarize the entire architecture across all parts."

    print(f"Specific Query: '{query_specific}'")
    vec_resp = vec_engine.query(query_specific)
    print(f"VectorStoreIndex Answer: {vec_resp}")

    print(f"\nBroad Query: '{query_broad}'")
    sum_resp = sum_engine.query(query_broad)
    print(f"SummaryIndex Answer: {sum_resp}")
    print("Exercise 3 passed successfully!")


def exercise_4_incremental_indexing():
    print("\n--- Exercise 4: Incremental Index Updates & Persistence ---")
    storage_path = Path(__file__).resolve().parent / "storage_exercise4"
    if storage_path.exists():
        shutil.rmtree(storage_path)

    # Initial document
    doc_initial = Document(text="Antigravity Release 1.0 supports automated tool calling.", metadata={"version": "1.0"})
    index = VectorStoreIndex.from_documents([doc_initial])
    index.storage_context.persist(persist_dir=str(storage_path))
    print(f"Persisted base index (v1.0) to {storage_path}")

    # Reload index
    storage_context = StorageContext.from_defaults(persist_dir=str(storage_path))
    reloaded_index = load_index_from_storage(storage_context)

    # Incremental update: insert doc 2
    doc_new = Document(text="Antigravity Release 2.0 adds multi-agent swarms and interactive human-in-the-loop.", metadata={"version": "2.0"})
    reloaded_index.insert(doc_new)
    print("Incrementally inserted doc (v2.0) into loaded index.")

    engine = reloaded_index.as_query_engine(similarity_top_k=2)
    resp = engine.query("What features were introduced in Release 2.0?")
    print(f"Query Result:\n{resp}")
    print("Exercise 4 passed successfully!")


def main():
    print("=" * 60)
    print("Module 10: LlamaIndex - Comprehensive Exercises")
    print("=" * 60)
    configure_llama_index()

    exercise_1_metadata_filtered_query()
    exercise_2_node_parser_analysis()
    exercise_3_vector_vs_summary_comparison()
    exercise_4_incremental_indexing()

    print("\n" + "=" * 60)
    print("All Module 10 Exercises Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
