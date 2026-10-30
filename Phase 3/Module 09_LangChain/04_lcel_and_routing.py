# Module 09 - LangChain
# 9.4 LCEL - LangChain Expression Language, Branching & Conditional Routing
#
# LCEL uses the | pipe operator to compose runnables into declarative workflows.
# Every component implements the Runnable interface (invoke, stream, batch).
# Topics covered:
#   1. Basic Chain: prompt | model | parser
#   2. Branching with RunnablePassthrough and RunnableParallel
#   3. Conditional routing with RunnableLambda / custom routers

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from utils import get_chat_model


def run_lcel_demo() -> None:
    print("=" * 60, flush=True)
    print("9.4 LCEL - Chains, Parallel Branching & Conditional Routing", flush=True)
    print("=" * 60, flush=True)

    # Mock responses for offline resilience
    mock_responses = [
        # Part 1 response
        "LangChain is a modular framework for building LLM applications using composable blocks. It simplifies prompts, models, and retrieval without boilerplate.",
        # Part 2 responses (summary, then keywords)
        "RAG enhances language model responses by retrieving relevant external context during inference.",
        "RAG, retrieval, context",
        # Part 3 responses (code, then prose)
        "def chunk_text(text: str, size: int = 100, overlap: int = 20) -> list[str]:\n    step = size - overlap\n    return [text[i:i + size] for i in range(0, len(text), step)]",
        "Attention is a mechanism that computes weighted importance scores between tokens in a sequence.",
    ]

    llm = get_chat_model(mock_responses=mock_responses)
    parser = StrOutputParser()

    # ── Part 1: Basic LCEL Chain (prompt | model | parser) ────────────────────
    print("\n--- Part 1: Basic LCEL Chain ---", flush=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarise the following text in exactly {sentences} sentences."),
        ("human",  "{text}"),
    ])
    chain = prompt | llm | parser

    text_input = {
        "sentences": 2,
        "text": """LangChain is a framework for developing applications powered by large language models.
It provides abstractions for prompts, models, memory, and retrieval so developers can
compose complex AI workflows from simple building blocks without writing boilerplate."""
    }

    try:
        result = chain.invoke(text_input)
        print("Summary Result:\n", result)
    except Exception as e:
        print(f"[API unreachable: {e}] Running offline fallback:")
        offline_chain = prompt | FakeListChatModel(responses=[mock_responses[0]]) | parser
        print(offline_chain.invoke(text_input))

    # ── Part 2: Branching with RunnableParallel & RunnablePassthrough ──────────
    print("\n--- Part 2: Parallel Branching (RunnableParallel) ---", flush=True)
    summarise_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarise this in one sentence."),
        ("human",  "{text}"),
    ])
    keywords_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract 3 keywords from this text. Return them comma-separated."),
        ("human",  "{text}"),
    ])

    parallel_chain = RunnableParallel(
        summary  = summarise_prompt | llm | parser,
        keywords = keywords_prompt  | llm | parser,
    )

    rag_text = {
        "text": "Retrieval-Augmented Generation (RAG) improves LLM answers by fetching relevant documents at inference time."
    }

    try:
        parallel_result = parallel_chain.invoke(rag_text)
        print("Summary: ", parallel_result["summary"])
        print("Keywords:", parallel_result["keywords"])
    except Exception as e:
        print(f"[API unreachable: {e}] Running offline parallel simulation:")
        offline_parallel = RunnableParallel(
            summary  = summarise_prompt | FakeListChatModel(responses=[mock_responses[1]]) | parser,
            keywords = keywords_prompt  | FakeListChatModel(responses=[mock_responses[2]]) | parser,
        )
        sim_res = offline_parallel.invoke(rag_text)
        print("Summary: ", sim_res["summary"])
        print("Keywords:", sim_res["keywords"])

    # ── Part 3: Conditional Routing ───────────────────────────────────────────
    print("\n--- Part 3: Conditional Routing with RunnableLambda ---", flush=True)
    code_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Python expert. Provide a complete, working code example."),
        ("human",  "{input}"),
    ])
    prose_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a clear technical writer. Explain conceptually."),
        ("human",  "{input}"),
    ])

    code_chain  = code_prompt  | llm | parser
    prose_chain = prose_prompt | llm | parser

    def route(input_dict: dict) -> str:
        """Route to code or prose chain based on the question."""
        q = input_dict["input"].lower()
        if any(w in q for w in ["write", "code", "implement", "example", "function"]):
            return "code"
        return "prose"

    def run_routed(input_dict: dict) -> str:
        choice = route(input_dict)
        try:
            if choice == "code":
                return code_chain.invoke(input_dict)
            return prose_chain.invoke(input_dict)
        except Exception:
            # Offline branch return
            if choice == "code":
                return FakeListChatModel(responses=[mock_responses[3]]).invoke(input_dict["input"]).content
            return FakeListChatModel(responses=[mock_responses[4]]).invoke(input_dict["input"]).content

    router_chain = RunnableLambda(run_routed)

    query_code = {"input": "Write a function to chunk text with overlap."}
    query_prose = {"input": "What is attention in transformers?"}

    print(f"\n[Routing query 1] -> '{query_code['input']}' (Route detected: {route(query_code)})")
    print(router_chain.invoke(query_code))

    print(f"\n[Routing query 2] -> '{query_prose['input']}' (Route detected: {route(query_prose)})")
    print(router_chain.invoke(query_prose))

    print("\n" + "=" * 60, flush=True)
    print("9.4 LCEL & Routing Demo Completed Successfully!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_lcel_demo()
