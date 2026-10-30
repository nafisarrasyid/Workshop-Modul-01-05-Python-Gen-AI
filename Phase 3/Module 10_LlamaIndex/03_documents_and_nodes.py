"""Module 10 - LlamaIndex
Section 10.3: Documents & Nodes

Demonstrates:
  - Creating Document objects with rich metadata
  - Splitting Documents into TextNode chunks using SentenceSplitter
  - Directly constructing TextNode objects with parent/child relationships
  - Inspecting node attributes, relationships, and metadata
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import configure_llama_index

from llama_index.core import Document
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.core.node_parser import SentenceSplitter

def main():
    print("=" * 60)
    print("Module 10.3: LlamaIndex Documents & Nodes")
    print("=" * 60)

    configure_llama_index()

    # 1. Create Document with Metadata
    sample_text = (
        "LlamaIndex is a data framework for LLM-based applications. "
        "It provides data connectors to ingest data from native sources like PDFs, APIs, and SQL. "
        "It indexes your data into structures that LLMs can easily query. "
        "Furthermore, LlamaIndex offers query engines and retrieval components for RAG pipelines."
    )
    doc = Document(
        text=sample_text,
        metadata={
            "source": "llamaindex_overview.txt",
            "category": "framework",
            "author": "LlamaIndex Team",
            "year": 2026,
        },
        doc_id="doc_rag_001",
    )
    print(f"Created Document ID: {doc.doc_id}")
    print(f"Metadata: {doc.metadata}")
    print(f"Text Snippet: {doc.text[:70]}...\n")

    # 2. Split Document into Nodes using SentenceSplitter
    splitter = SentenceSplitter(chunk_size=40, chunk_overlap=10)
    nodes = splitter.get_nodes_from_documents([doc])

    print(f"Split document into {len(nodes)} TextNode chunks:")
    for idx, node in enumerate(nodes, start=1):
        print(f"  [Node {idx}] ID: {node.node_id}")
        print(f"    Text: {node.text}")
        print(f"    Ref Doc ID: {node.ref_doc_id}")
        print(f"    Inherited Meta: {node.metadata}")

    # 3. Explicitly construct custom TextNode with relationships
    parent_node = TextNode(
        text="Parent Concept: Retrieval-Augmented Generation Architecture",
        id_="node_parent_100",
        metadata={"level": "parent"},
    )
    child_node = TextNode(
        text="Child Detail: Dense vector search retrieves top-k relevant nodes.",
        id_="node_child_101",
        metadata={"level": "child"},
        relationships={
            NodeRelationship.PARENT: RelatedNodeInfo(node_id=parent_node.node_id)
        },
    )
    print("\nCustom Relationship Node:")
    print(f"  Child Node ID: {child_node.node_id}")
    print(f"  Parent Reference: {child_node.relationships[NodeRelationship.PARENT].node_id}")

    print("\nDocument and Node workflows completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
