"""Module 11 - Vector Databases
Section 11.3: Pinecone

Demonstrates:
  - Pinecone client initialization (v7 SDK)
  - Index management & ServerlessSpec configuration
  - Upserting vectors with namespaces and structured metadata
  - Executing namespace-isolated vector queries with metadata filtering
"""
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import DeterministicMockEmbeddings, check_host_reachable

def get_pinecone_client():
    api_key = os.getenv("PINECONE_API_KEY")
    if api_key and api_key != "your_pinecone_api_key_here":
        try:
            from pinecone import Pinecone
            return Pinecone(api_key=api_key), False
        except Exception:
            pass

    # High-fidelity Pinecone Mock Client for local/offline execution
    class MockPineconeIndex:
        def __init__(self, name: str):
            self.name = name
            # namespace -> list of {id, values, metadata}
            self.namespaces = {}

        def upsert(self, vectors, namespace=""):
            if namespace not in self.namespaces:
                self.namespaces[namespace] = {}
            for v in vectors:
                self.namespaces[namespace][v["id"]] = v
            return {"upserted_count": len(vectors)}

        def query(self, vector, top_k=3, namespace="", include_metadata=True, filter=None):
            ns_data = self.namespaces.get(namespace, {})
            candidates = list(ns_data.values())

            # Apply metadata filter
            if filter:
                def match(meta, f_dict):
                    for k, condition in f_dict.items():
                        if isinstance(condition, dict):
                            for op, target in condition.items():
                                val = meta.get(k)
                                if op == "$eq" and val != target: return False
                                if op == "$ne" and val == target: return False
                                if op == "$gte" and (val is None or val < target): return False
                                if op == "$lte" and (val is None or val > target): return False
                                if op == "$in" and val not in target: return False
                        elif meta.get(k) != condition:
                            return False
                    return True
                candidates = [c for c in candidates if match(c.get("metadata", {}), filter)]

            # Compute cosine similarity
            q_vec = np.array(vector, dtype=float)
            q_norm = np.linalg.norm(q_vec) or 1.0
            scored = []
            for c in candidates:
                c_vec = np.array(c["values"], dtype=float)
                c_norm = np.linalg.norm(c_vec) or 1.0
                score = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
                scored.append((score, c))

            scored.sort(key=lambda x: x[0], reverse=True)
            matches = []
            for score, c in scored[:top_k]:
                item = {"id": c["id"], "score": score}
                if include_metadata:
                    item["metadata"] = c.get("metadata", {})
                matches.append(item)
            return {"matches": matches, "namespace": namespace}

    class MockPinecone:
        def __init__(self):
            self._indexes = {}

        def list_indexes(self):
            class IndexList:
                def names(self): return list(self._indexes.keys())
            return IndexList()

        def create_index(self, name, dimension, metric="cosine", spec=None):
            self._indexes[name] = MockPineconeIndex(name)

        def Index(self, name):
            if name not in self._indexes:
                self._indexes[name] = MockPineconeIndex(name)
            return self._indexes[name]

    return MockPinecone(), True


def main():
    print("=" * 60)
    print("Module 11.3: Pinecone Cloud Vector Storage")
    print("=" * 60)

    pc, is_mock = get_pinecone_client()
    mode_str = "Simulated / Local Mode" if is_mock else "Cloud Connected"
    print(f"Pinecone Status: {mode_str}")

    index_name = "genai-docs"
    # Ensure index is created
    try:
        from pinecone import ServerlessSpec
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
    except Exception:
        spec = None

    try:
        pc.create_index(name=index_name, dimension=32, metric="cosine", spec=spec)
    except Exception:
        pass  # Already exists

    index = pc.Index(index_name)
    print(f"Connected to Pinecone Index: '{index_name}'")

    embedder = DeterministicMockEmbeddings(dim=32)

    # 1. Prepare vectors for two separate namespaces: "engineering" and "research"
    eng_docs = [
        {"id": "eng_1", "text": "Continuous Integration with GitHub Actions and Docker build caching.", "env": "prod", "year": 2024},
        {"id": "eng_2", "text": "Kubernetes horizontal pod autoscaling based on custom Prometheus metrics.", "env": "prod", "year": 2025},
    ]
    res_docs = [
        {"id": "res_1", "text": "Diffusion models for text-to-3D asset generation using score distillation.", "env": "research", "year": 2024},
        {"id": "res_2", "text": "Mechanistic interpretability of induction heads in transformer circuits.", "env": "research", "year": 2025},
    ]

    # Upsert into "engineering" namespace
    eng_vectors = [
        {"id": d["id"], "values": embedder.embed_query(d["text"]), "metadata": {"text": d["text"], "env": d["env"], "year": d["year"]}}
        for d in eng_docs
    ]
    index.upsert(vectors=eng_vectors, namespace="engineering")
    print(f"Upserted {len(eng_vectors)} vectors into namespace 'engineering'")

    # Upsert into "research" namespace
    res_vectors = [
        {"id": d["id"], "values": embedder.embed_query(d["text"]), "metadata": {"text": d["text"], "env": d["env"], "year": d["year"]}}
        for d in res_docs
    ]
    index.upsert(vectors=res_vectors, namespace="research")
    print(f"Upserted {len(res_vectors)} vectors into namespace 'research'")

    # 2. Query within specific namespace with metadata filter
    query_text = "Docker container deployments and orchestration"
    q_vec = embedder.embed_query(query_text)

    print(f"\nQuerying: '{query_text}'")
    print("Target Namespace: 'engineering' | Filter: {'year': {'$gte': 2024}}")

    query_res = index.query(
        vector=q_vec,
        top_k=2,
        namespace="engineering",
        include_metadata=True,
        filter={"year": {"$gte": 2024}},
    )

    print(f"\nMatches in namespace '{query_res.get('namespace')}':")
    for match in query_res.get("matches", []):
        print(f"  [ID: {match['id']}] Score: {match['score']:.4f}")
        print(f"    Text: {match['metadata'].get('text')}")
        print(f"    Year: {match['metadata'].get('year')}")

    print("\nPinecone workflow completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
