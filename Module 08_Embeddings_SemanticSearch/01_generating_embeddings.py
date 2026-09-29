# Module 08 - Embeddings & Semantic Search
# 8.1 What Are Embeddings?
# An embedding is a fixed-length vector of floating-point numbers that
# represents a piece of text. Semantically similar texts produce vectors that
# are geometrically close. This property powers semantic search, clustering,
# deduplication, and the retrieval step in RAG.
#
#   "What is RAG?"                    -> [0.021, -0.143, 0.892, ...]  (1536 numbers)
#   "Retrieval Augmented Generation"  -> [0.019, -0.139, 0.881, ...]  (geometrically close)
#   "Who won the cricket match?"      -> [-0.312, 0.401, -0.203, ...] (geometrically far)
#
# Key properties:
# - Dimensionality: 768 to 3072 dimensions depending on the model
# - Magnitude: vectors are usually L2-normalised (length = 1)
# - Comparison: cosine similarity is the standard metric
#
# 8.2 Generating Embeddings
#
# NOTE: embed() below makes a REAL, billed API call to OpenAI. It requires a
# valid OPENAI_API_KEY in a .env file in this folder to run.

from openai import OpenAI
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def embed(texts: list[str], model: str = None) -> np.ndarray:
    """Embed a list of texts. Returns array of shape (n, dim)."""
    # Use Gemini embedding if available in .env, otherwise fallback to OpenAI / OmniRoute
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OMNIROUTE_API_KEY")
    base_url = os.getenv("GEMINI_OPENAI_BASE_URL")
    if not os.getenv("GEMINI_API_KEY"):
        base_url = os.getenv("OMNIROUTE_BASE_URL")

    model_name = model or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    client = OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
    )
    # API accepts up to 2048 texts per call
    response = client.embeddings.create(input=texts, model=model_name)
    # Handle sorting safely if index is present, or preserve order
    if all(hasattr(e, "index") and e.index is not None for e in response.data):
        vectors = sorted(response.data, key=lambda e: e.index)
    else:
        vectors = response.data
    return np.array([v.embedding for v in vectors], dtype=np.float32)


TEXTS = [
    "Retrieval-Augmented Generation combines search with LLMs.",
    "RAG retrieves documents then generates an answer from them.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Fine-tuning trains a model on new data.",
]


if __name__ == "__main__":
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OMNIROUTE_API_KEY")):
        print("GEMINI_API_KEY / OPENAI_API_KEY / OMNIROUTE_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        embeddings = embed(TEXTS)
        print(f"Shape: {embeddings.shape}")
        print(f"Norm of first vector: {np.linalg.norm(embeddings[0]):.4f}")

    # NOTE on Voyage AI (Anthropic's recommended embedding provider):
    # pip install voyageai, then:
    #
    # import voyageai
    # vo = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
    # result = vo.embed(
    #     ["What is RAG?", "Explain vector databases."],
    #     model="voyage-3",
    #     input_type="document",   # "document" for corpus, "query" for search queries
    # )
    # embeddings = np.array(result.embeddings, dtype=np.float32)
    #
    # Use text-embedding-3-small if you are already on OpenAI; use voyage-3
    # for Anthropic/Claude stacks. Both are excellent general-purpose choices.
