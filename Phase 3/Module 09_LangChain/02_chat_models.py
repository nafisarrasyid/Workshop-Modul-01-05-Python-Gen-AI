# Module 09 - LangChain
# 9.2 Chat Models
#
# LangChain provides unified abstractions over LLM providers (OpenAI, Anthropic, etc.).
# Every chat model supports three standard execution modes:
#   1. invoke() - Synchronous blocking call
#   2. stream() - Incremental token streaming
#   3. batch()  - Parallel execution over multiple message lists

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from utils import get_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def run_chat_models_demo() -> None:
    print("=" * 60, flush=True)
    print("9.2 Chat Models - LangChain Model Interface Demo", flush=True)
    print("=" * 60, flush=True)

    # 1. Configuration & Model Initialization
    # Uses real OpenAI/Anthropic/OmniRoute model when online, or simulated ChatModel
    llm = get_chat_model(
        mock_responses=[
            "LangChain Expression Language (LCEL) is a declarative, pipe-based syntax that makes composition explicit and readable.",
            "Retrieval-Augmented Generation (RAG) grounds LLMs with external documents.",
            "A vector database stores embeddings and enables semantic nearest-neighbor search.",
        ]
    )

    # 2. Both accept the exact same message types
    messages = [
        SystemMessage(content="You are a concise technical assistant."),
        HumanMessage(content="What is LangChain Expression Language?"),
    ]

    print("\n--- 1. Invoke (Blocking) ---", flush=True)
    try:
        response = llm.invoke(messages)
        print(response.content, flush=True)
    except Exception as e:
        print(f"[Fallback]: LCEL makes chain composition explicit and readable.")

    print("\n--- 2. Stream (Token by Token) ---", flush=True)
    try:
        for chunk in llm.stream(messages):
            print(chunk.content, end="", flush=True)
        print(flush=True)
    except Exception:
        print("[Offline Stream]: LCEL provides seamless streaming support out of the box.", flush=True)

    print("\n--- 3. Batch (Parallel Execution) ---", flush=True)
    batch_prompts = [
        [HumanMessage(content="What is RAG?")],
        [HumanMessage(content="What is a vector database?")],
    ]
    try:
        responses = llm.batch(batch_prompts)
        for i, r in enumerate(responses, 1):
            print(f"Batch item {i}: {r.content[:100]}...", flush=True)
    except Exception:
        fallback_batch = [
            "RAG (Retrieval-Augmented Generation) combines document retrieval with text generation.",
            "A vector database is an optimized storage engine for vector embeddings."
        ]
        for i, r in enumerate(fallback_batch, 1):
            print(f"Batch item {i}: {r}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("9.2 Chat Models Demo Completed Successfully!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_chat_models_demo()
