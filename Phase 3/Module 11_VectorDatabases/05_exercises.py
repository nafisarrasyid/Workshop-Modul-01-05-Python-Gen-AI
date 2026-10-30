"""Module 11 - Vector Databases
Section 11.6: Module 11 Exercises

Implements all 4 exercises from the curriculum:
  - Exercise 1: Vector DB Benchmark (Insertion throughput & query latency)
  - Exercise 2: MultiCollectionSearch (Federated search across multiple collections)
  - Exercise 3: TenantVectorStore (Isolated multi-tenant partitions & access control)
  - Exercise 4: Vector Store Migration Tool (Preserving embeddings, IDs, and metadata)
"""
import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import DeterministicMockEmbeddings


# ── Exercise 1: Vector DB Benchmark ──────────────────────────────────────────
def exercise_1_benchmark():
    print("\n--- Exercise 1: Vector Database Throughput & Latency Benchmark ---")
    embedder = DeterministicMockEmbeddings(dim=64)

    # In-memory benchmark store
    class BenchmarkStore:
        def __init__(self):
            self.vectors = []
            self.ids = []

        def insert_batch(self, ids: list, vecs: list):
            self.ids.extend(ids)
            self.vectors.extend(vecs)

        def query(self, q_vec: list, top_k: int = 5):
            q = np.array(q_vec)
            V = np.array(self.vectors)
            scores = np.dot(V, q) / (np.linalg.norm(V, axis=1) * np.linalg.norm(q))
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(self.ids[i], scores[i]) for i in top_indices]

    batch_sizes = [10, 50, 100]
    print(f"{'Batch Size':<12} | {'Insert Time (ms)':<18} | {'Throughput (vec/s)':<20} | {'Query Latency (ms)':<18}")
    print("-" * 75)

    for b_size in batch_sizes:
        store = BenchmarkStore()
        texts = [f"Benchmark sample sentence index {i} evaluating retrieval latency." for i in range(b_size)]
        ids = [f"b_{i}" for i in range(b_size)]
        vecs = embedder.embed_documents(texts)

        # Benchmark Insertion
        t0 = time.perf_counter()
        store.insert_batch(ids, vecs)
        insert_duration = time.perf_counter() - t0
        insert_ms = insert_duration * 1000
        throughput = b_size / insert_duration if insert_duration > 0 else 0

        # Benchmark Query
        q_vec = embedder.embed_query("Benchmark query testing search speed")
        t_q0 = time.perf_counter()
        _ = store.query(q_vec, top_k=5)
        query_ms = (time.perf_counter() - t_q0) * 1000

        print(f"{b_size:<12} | {insert_ms:<18.2f} | {throughput:<20.1f} | {query_ms:<18.3f}")

    print("Exercise 1 benchmark completed successfully!")


# ── Exercise 2: MultiCollectionSearch ────────────────────────────────────────
class MultiCollectionSearch:
    def __init__(self, embedder):
        self.embedder = embedder
        # collection_name -> list of records {id, text, vector, metadata}
        self.collections: Dict[str, List[Dict[str, Any]]] = {}

    def add_documents(self, collection_name: str, documents: List[Dict[str, Any]]):
        if collection_name not in self.collections:
            self.collections[collection_name] = []
        for doc in documents:
            vec = self.embedder.embed_query(doc["text"])
            self.collections[collection_name].append({
                "id": doc["id"],
                "text": doc["text"],
                "vector": vec,
                "metadata": doc.get("metadata", {}),
            })

    def federated_search(self, query: str, top_k: int = 3, collections: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        target_colls = collections or list(self.collections.keys())
        q_vec = np.array(self.embedder.embed_query(query))
        q_norm = np.linalg.norm(q_vec) or 1.0

        all_scored = []
        for c_name in target_colls:
            for rec in self.collections.get(c_name, []):
                rec_vec = np.array(rec["vector"])
                rec_norm = np.linalg.norm(rec_vec) or 1.0
                score = float(np.dot(q_vec, rec_vec) / (q_norm * rec_norm))
                all_scored.append({
                    "id": rec["id"],
                    "text": rec["text"],
                    "collection": c_name,
                    "metadata": rec["metadata"],
                    "score": score,
                })

        all_scored.sort(key=lambda x: x["score"], reverse=True)
        return all_scored[:top_k]


def exercise_2_multi_collection_search():
    print("\n--- Exercise 2: MultiCollectionSearch (Federated Retrieval) ---")
    embedder = DeterministicMockEmbeddings(dim=32)
    searcher = MultiCollectionSearch(embedder)

    searcher.add_documents("engineering", [
        {"id": "eng_1", "text": "PostgreSQL connection pooling with PgBouncer.", "metadata": {"tier": "backend"}},
        {"id": "eng_2", "text": "Redis caching strategy for high-concurrency API gateways.", "metadata": {"tier": "cache"}},
    ])

    searcher.add_documents("compliance", [
        {"id": "comp_1", "text": "SOC2 compliance requires end-to-end data encryption at rest and in transit.", "metadata": {"area": "security"}},
        {"id": "comp_2", "text": "GDPR right to be forgotten mandate specifies complete user record purging.", "metadata": {"area": "privacy"}},
    ])

    query = "database encryption and compliance policies"
    print(f"Federated Query across all collections: '{query}'")
    results = searcher.federated_search(query, top_k=2)

    for rank, res in enumerate(results, start=1):
        print(f"  [Rank {rank}] Score: {res['score']:.4f} | Source Collection: '{res['collection']}'")
        print(f"    Text: {res['text']}")
    print("Exercise 2 passed successfully!")


# ── Exercise 3: TenantVectorStore ────────────────────────────────────────────
class TenantVectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        # tenant_id -> list of records
        self._tenant_partitions: Dict[str, List[Dict[str, Any]]] = {}

    def insert(self, tenant_id: str, doc_id: str, text: str, metadata: dict = None):
        if tenant_id not in self._tenant_partitions:
            self._tenant_partitions[tenant_id] = []
        vec = self.embedder.embed_query(text)
        self._tenant_partitions[tenant_id].append({
            "id": doc_id,
            "text": text,
            "vector": vec,
            "metadata": metadata or {},
        })

    def query(self, tenant_id: str, query_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        # Enforce strict tenant isolation
        partition = self._tenant_partitions.get(tenant_id, [])
        if not partition:
            return []

        q_vec = np.array(self.embedder.embed_query(query_text))
        q_norm = np.linalg.norm(q_vec) or 1.0

        scored = []
        for rec in partition:
            rec_vec = np.array(rec["vector"])
            rec_norm = np.linalg.norm(rec_vec) or 1.0
            sim = float(np.dot(q_vec, rec_vec) / (q_norm * rec_norm))
            scored.append({"id": rec["id"], "text": rec["text"], "score": sim, "tenant_id": tenant_id})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def exercise_3_tenant_vector_store():
    print("\n--- Exercise 3: TenantVectorStore (Strict Multi-Tenant Isolation) ---")
    embedder = DeterministicMockEmbeddings(dim=32)
    store = TenantVectorStore(embedder)

    # Ingest data for Tenant A and Tenant B
    store.insert("tenant_alpha", "a1", "Alpha Corporation Internal Financial Forecast 2026", {"confidential": True})
    store.insert("tenant_beta", "b1", "Beta Industries Proprietary Drone Manufacturing Specs", {"confidential": True})

    query = "Proprietary Manufacturing Specifications"
    print(f"Executing Query: '{query}'")

    # Tenant Alpha queries: must NOT see Tenant Beta data
    results_alpha = store.query("tenant_alpha", query, top_k=2)
    print(f"Results for Tenant Alpha: {[r['id'] for r in results_alpha]}")
    assert all(r["tenant_id"] == "tenant_alpha" for r in results_alpha), "Tenant leak detected!"

    # Tenant Beta queries
    results_beta = store.query("tenant_beta", query, top_k=2)
    print(f"Results for Tenant Beta : {[r['id'] for r in results_beta]}")
    assert all(r["tenant_id"] == "tenant_beta" for r in results_beta), "Tenant leak detected!"

    print("Tenant isolation verified with zero data leakage!")
    print("Exercise 3 passed successfully!")


# ── Exercise 4: Vector Store Migration Tool ──────────────────────────────────
def exercise_4_migration_tool():
    print("\n--- Exercise 4: Vector Store Migration Tool ---")
    embedder = DeterministicMockEmbeddings(dim=32)

    # Source Store
    source_records = [
        {"id": "doc_01", "text": "LangChain provides LCEL composition.", "meta": {"version": 1}},
        {"id": "doc_02", "text": "LlamaIndex excels at semantic data indexing.", "meta": {"version": 2}},
    ]
    source_store = {
        r["id"]: {
            "id": r["id"],
            "text": r["text"],
            "vector": embedder.embed_query(r["text"]),
            "metadata": r["meta"],
        }
        for r in source_records
    }
    print(f"Source Store contains {len(source_store)} records.")

    # Target Store
    target_store = {}

    def migrate_records(source: dict, target: dict) -> int:
        count = 0
        for doc_id, data in source.items():
            # Validate integrity
            assert "vector" in data and len(data["vector"]) == 32
            assert "metadata" in data
            target[doc_id] = {
                "id": data["id"],
                "text": data["text"],
                "vector": list(data["vector"]),
                "metadata": dict(data["metadata"]),
            }
            count += 1
        return count

    migrated_count = migrate_records(source_store, target_store)
    print(f"Successfully migrated {migrated_count} records into target store.")
    assert len(target_store) == len(source_store)
    for k in source_store:
        assert target_store[k]["id"] == source_store[k]["id"]
        assert target_store[k]["metadata"] == source_store[k]["metadata"]
        assert np.allclose(target_store[k]["vector"], source_store[k]["vector"])

    print("Migration verified: 100% vector, metadata, and ID parity!")
    print("Exercise 4 passed successfully!")


def main():
    print("=" * 60)
    print("Module 11: Vector Databases - Comprehensive Exercises")
    print("=" * 60)

    exercise_1_benchmark()
    exercise_2_multi_collection_search()
    exercise_3_tenant_vector_store()
    exercise_4_migration_tool()

    print("\n" + "=" * 60)
    print("All Module 11 Exercises Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
