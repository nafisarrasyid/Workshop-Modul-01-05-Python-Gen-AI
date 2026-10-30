# Module 09 - LangChain
# 9.3 Prompt Templates
#
# Prompt templates decouple static prompt structure from dynamic input variables.
# Key primitives:
#   1. ChatPromptTemplate - Composes system and human message schemas with {variables}.
#   2. MessagesPlaceholder - Injects variable-length conversation histories dynamically.

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from utils import get_chat_model


def run_prompt_templates_demo() -> None:
    print("=" * 60, flush=True)
    print("9.3 Prompt Templates - ChatPromptTemplate & MessagesPlaceholder", flush=True)
    print("=" * 60, flush=True)

    # ── 1. Simple Template ────────────────────────────────────────────────────
    print("\n--- 1. Simple ChatPromptTemplate ---", flush=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a {role}. Answer in {language}."),
        ("human",  "{question}"),
    ])

    # Format to messages (for inspection)
    msgs = prompt.format_messages(
        role="Python tutor",
        language="English",
        question="What is a generator?",
    )

    print("Formatted messages list:")
    for m in msgs:
        print(f"  [{m.type.upper()}]: {m.content}")

    # ── 2. Template with Conversation History Placeholder ────────────────────
    print("\n--- 2. Template with MessagesPlaceholder ---", flush=True)
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Today's context: {context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human",  "{input}"),
    ])

    formatted = chat_prompt.format_messages(
        context="User is learning about RAG",
        history=[
            HumanMessage(content="What is retrieval?"),
        ],
        input="How is it different from fine-tuning?",
    )

    print(f"Total messages in formatted prompt: {len(formatted)}")
    for i, m in enumerate(formatted, 1):
        print(f"  {i}. [{m.type.upper()}]: {m.content}")

    # ── 3. Execution Pipeline (Prompt -> Model -> Parser) ────────────────────
    print("\n--- 3. Running Chain with PromptTemplate ---", flush=True)
    mock_answer = (
        "In Python, a generator is a function that produces a sequence of values "
        "on demand using the 'yield' statement, conserving memory compared to lists."
    )
    llm = get_chat_model(mock_responses=[mock_answer])
    chain = prompt | llm | StrOutputParser()

    input_payload = {
        "role": "concise Python tutor",
        "language": "English",
        "question": "What is a generator in one sentence?",
    }

    try:
        response = chain.invoke(input_payload)
        print("Model Response:\n", response)
    except Exception as e:
        print(f"[API unreachable: {e}] Running offline simulation:")
        fallback_chain = prompt | FakeListChatModel(responses=[mock_answer]) | StrOutputParser()
        print("Model Response (offline):\n", fallback_chain.invoke(input_payload))

    print("\n" + "=" * 60, flush=True)
    print("9.3 Prompt Templates Demo Completed Successfully!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_prompt_templates_demo()
