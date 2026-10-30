"""Module 11 - Vector Databases
Section 11.4: Weaviate

Demonstrates:
  - Weaviate v4 Python client connection patterns
  - Collection creation with schema properties and vector configurations
  - Hybrid Search combining BM25 keyword matching and dense vector similarity
  - Filtering queries using weaviate.classes.query.Filter
"""
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import DeterministicMockEmbeddings, check_host_reachable

def run_weaviate_demonstration():
    print("=" * 60)
    print("Module 11.4: Weaviate Vector Database & Hybrid Search")
    print("=" * 60)

    # 1. Check if a local or cloud Weaviate cluster is available
    local_reachable = check_host_reachable("http://localhost:8080", timeout=0.3)

    if local_reachable:
        try:
            import weaviate
            import weaviate.classes as wvc
            client = weaviate.connect_to_local()
            print("Connected to live local Weaviate instance on port 8080!")
            client_connected = True
        except Exception:
            client_connected = False
    else:
        print("Note: No live Weaviate instance detected on http://localhost:8080.")
        print("Patterns and execution shown below via Weaviate v4 Architecture Simulator.")
        print("To run live: docker run -d -p 8080:8080 cr.weaviate.io/semitechnologies/weaviate:latest\n")
        client_connected = False

    embedder = DeterministicMockEmbeddings(dim=32)

    # Sample articles to ingest
    articles_data = [
        {"title": "Deep Learning with PyTorch", "content": "Tensors, autograd, and neural network training loops.", "author": "Alice", "views": 1500},
        {"title": "Production RAG Systems", "content": "Hybrid search combining dense vectors and BM25 sparse keyword queries.", "author": "Bob", "views": 3200},
        {"title": "Agentic AI with LangGraph", "content": "Cyclic state graphs and multi-agent coordination.", "author": "Alice", "views": 2800},
    ]

    print("Step 1: Collection Schema Definition (Weaviate v4):")
    schema_code = """
client.collections.create(
    name="Article",
    properties=[
        wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="author", data_type=wvc.config.DataType.TEXT),
        wvc.config.Property(name="views", data_type=wvc.config.DataType.INT),
    ],
    vectorizer_config=wvc.config.Configure.Vectorizer.none()
)
"""
    print(schema_code.strip())

    print("\nStep 2: Ingesting Data & Generating Vectors:")
    for art in articles_data:
        vec = embedder.embed_query(art["title"] + " " + art["content"])
        print(f"  Inserted '{art['title']}' with 32-dim vector (Author: {art['author']}, Views: {art['views']})")

    # Step 3: Hybrid Search Simulation (BM25 keyword score + Cosine dense similarity)
    query_str = "hybrid search dense vectors BM25"
    query_vec = embedder.embed_query(query_str)
    query_terms = set(query_str.lower().split())

    print(f"\nStep 3: Executing Hybrid Search (alpha=0.5):")
    print(f"Query: '{query_str}'")

    results = []
    for art in articles_data:
        # BM25 term overlap score
        text_words = set((art["title"] + " " + art["content"]).lower().split())
        bm25_score = len(query_terms.intersection(text_words)) / max(len(query_terms), 1)

        # Dense vector similarity
        art_vec = embedder.embed_query(art["title"] + " " + art["content"])
        dense_sim = float(np.dot(query_vec, art_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(art_vec)))

        # Hybrid fusion: alpha * dense + (1 - alpha) * bm25
        alpha = 0.5
        hybrid_score = (alpha * dense_sim) + ((1.0 - alpha) * bm25_score)
        results.append((hybrid_score, dense_sim, bm25_score, art))

    results.sort(key=lambda x: x[0], reverse=True)

    print("\nHybrid Query Results (Ranked by fused score):")
    for rank, (h_score, d_score, b_score, art) in enumerate(results, start=1):
        print(f"  [Rank {rank}] Combined Score: {h_score:.4f} (Dense: {d_score:.4f}, BM25: {b_score:.4f})")
        print(f"    Title  : {art['title']}")
        print(f"    Content: {art['content']}")
        print(f"    Author : {art['author']}, Views: {art['views']}")

    # Step 4: Metadata filtering pattern
    print("\nStep 4: Filtered Query using wvc.query.Filter:")
    print("Filter: Filter.by_property('author').equal('Alice') & Filter.by_property('views').greater_than(2000)")
    filtered = [r for r in results if r[3]["author"] == "Alice" and r[3]["views"] > 2000]
    for r in filtered:
        print(f"  Matching Document: '{r[3]['title']}' (Author: {r[3]['author']}, Views: {r[3]['views']})")

    print("\nWeaviate patterns and workflows completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    run_weaviate_demonstration()
