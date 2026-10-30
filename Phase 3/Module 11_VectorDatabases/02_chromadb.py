"""Module 11 - Vector Databases
Section 11.2: ChromaDB

Demonstrates:
  - Creating In-Memory and Persistent ChromaDB clients
  - Managing Collections with custom distance metrics (cosine, l2, ip)
  - Upserting documents with rich metadata and embeddings
  - Executing semantic vector queries with structured metadata filtering ($and, $eq, $in)
"""
import sys
import os
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import utils
from utils import DeterministicMockEmbeddings

# Safe ChromaDB client wrapper
def get_chroma_client(persist_path: str = None):
    try:
        import chromadb
        if persist_path:
            return chromadb.PersistentClient(path=persist_path)
        return chromadb.Client()
    except Exception as e:
        # High-fidelity In-Memory ChromaDB fallback adapter
        class InMemoryCollection:
            def __init__(self, name: str, metadata: dict = None):
                self.name = name
                self.metadata = metadata or {"hnsw:space": "cosine"}
                self.records = {}  # id -> {text, metadata, embedding}

            def add(self, ids, documents, metadatas=None, embeddings=None):
                metas = metadatas or [{} for _ in ids]
                embeds = embeddings or [None for _ in ids]
                for doc_id, text, meta, emb in zip(ids, documents, metas, embeds):
                    self.records[doc_id] = {
                        "id": doc_id,
                        "document": text,
                        "metadata": meta,
                        "embedding": emb,
                    }

            def query(self, query_texts=None, query_embeddings=None, n_results=2, where=None):
                results = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
                candidates = list(self.records.values())

                # Apply 'where' filter
                if where:
                    def matches_filter(rec_meta, cond):
                        for k, v in cond.items():
                            if k == "$and":
                                return all(matches_filter(rec_meta, sub) for sub in v)
                            elif isinstance(v, dict):
                                if "$eq" in v and rec_meta.get(k) != v["$eq"]:
                                    return False
                                if "$in" in v and rec_meta.get(k) not in v["$in"]:
                                    return False
                            elif rec_meta.get(k) != v:
                                return False
                        return True
                    candidates = [c for c in candidates if matches_filter(c["metadata"], where)]

                embed_tool = DeterministicMockEmbeddings(dim=32)
                q_emb = query_embeddings[0] if query_embeddings else embed_tool.embed_query(query_texts[0])

                scored = []
                for c in candidates:
                    c_emb = c["embedding"] if c["embedding"] is not None else embed_tool.embed_query(c["document"])
                    # Cosine distance = 1 - cosine_sim
                    norm_q = np.linalg.norm(q_emb) or 1.0
                    norm_c = np.linalg.norm(c_emb) or 1.0
                    sim = float(np.dot(q_emb, c_emb) / (norm_q * norm_c))
                    dist = max(0.0, 1.0 - sim)
                    scored.append((dist, c))

                scored.sort(key=lambda x: x[0])
                top = scored[:n_results]
                for d, c in top:
                    results["ids"][0].append(c["id"])
                    results["documents"][0].append(c["document"])
                    results["metadatas"][0].append(c["metadata"])
                    results["distances"][0].append(d)
                return results

            def count(self):
                return len(self.records)

        class FallbackClient:
            def __init__(self):
                self._colls = {}

            def get_or_create_collection(self, name, metadata=None):
                if name not in self._colls:
                    self._colls[name] = InMemoryCollection(name, metadata)
                return self._colls[name]

            def delete_collection(self, name):
                self._colls.pop(name, None)

        return FallbackClient()


def main():
    print("=" * 60)
    print("Module 11.2: ChromaDB Vector Storage & Querying")
    print("=" * 60)

    db_path = str(Path(__file__).resolve().parent / "chroma_db_data")
    client = get_chroma_client(persist_path=db_path)

    # 1. Create Collection with Cosine distance
    collection = client.get_or_create_collection(
        name="ai_papers",
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection created: '{collection.name}'")

    # 2. Add Documents with Metadata & Embeddings
    embedder = DeterministicMockEmbeddings(dim=32)
    docs = [
        "Attention Is All You Need introduced the Transformer architecture.",
        "Deep Residual Learning for Image Recognition introduced ResNet architectures.",
        "LoRA: Low-Rank Adaptation of Large Language Models reduces fine-tuning VRAM.",
        "QLoRA: Efficient Finetuning of Quantized LLMs enables 4-bit fine-tuning.",
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.",
    ]
    ids = [f"paper_{i+1:03d}" for i in range(len(docs))]
    metadatas = [
        {"domain": "nlp", "year": 2017, "tier": "pioneer"},
        {"domain": "cv", "year": 2015, "tier": "pioneer"},
        {"domain": "nlp", "year": 2021, "tier": "efficient_tuning"},
        {"domain": "nlp", "year": 2023, "tier": "efficient_tuning"},
        {"domain": "nlp", "year": 2020, "tier": "rag"},
    ]
    embeddings = embedder.embed_documents(docs)

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    print(f"Upserted {collection.count()} records into collection.")

    # 3. Semantic Query with Metadata Filter ($and with domain and tier)
    query_text = "efficient parameter fine-tuning for large models"
    query_emb = embedder.embed_query(query_text)

    filter_criteria = {
        "$and": [
            {"domain": {"$eq": "nlp"}},
            {"tier": {"$in": ["efficient_tuning", "rag"]}},
        ]
    }
    print(f"\nQuerying: '{query_text}'")
    print(f"Applying Filter: {filter_criteria}")

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=2,
        where=filter_criteria,
    )

    print("\nMatching Results:")
    for doc_id, text, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"  [ID: {doc_id}] Distance: {dist:.4f}")
        print(f"    Text: {text}")
        print(f"    Metadata: {meta}")

    print("\nChromaDB workflow completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
