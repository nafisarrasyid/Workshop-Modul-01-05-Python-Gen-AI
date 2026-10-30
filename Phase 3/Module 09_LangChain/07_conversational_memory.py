# Module 09 - LangChain
# 9.7 Conversational Memory
#
# LangChain manages multi-turn state across stateless LLM calls via:
#   1. InMemoryChatMessageHistory - Thread-safe session message storage
#   2. RunnableWithMessageHistory - Wrapper that automatically appends user and assistant turns
#   3. MessagesPlaceholder - Injects past turns into the system prompt

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from utils import get_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.fake_chat_models import FakeListChatModel


def run_conversational_memory_demo() -> None:
    print("=" * 60, flush=True)
    print("9.7 Conversational Memory - RunnableWithMessageHistory Demo", flush=True)
    print("=" * 60, flush=True)

    # 1. Base Chain Setup
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI tutor. Be concise."),
        MessagesPlaceholder(variable_name="history"),
        ("human",  "{input}"),
    ])
    parser = StrOutputParser()

    mock_answers = [
        "RAG (Retrieval-Augmented Generation) connects LLMs to external data by retrieving relevant chunks.",
        "You asked about RAG, which consists of retrieval and generation components.",
    ]

    llm = get_chat_model(mock_responses=mock_answers)
    base_chain = prompt | llm | parser

    # 2. Session Store keyed by session_id
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # 3. Wrap chain with history management
    chain_with_history = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "user-123"}}

    # Turn 1
    print("\n--- Turn 1 ---", flush=True)
    q1 = "What is RAG?"
    print(f"User: {q1}")
    try:
        r1 = chain_with_history.invoke({"input": q1}, config=config)
        print(f"Assistant: {r1}\n")
    except Exception as e:
        print(f"[API unreachable: {e}] Running offline simulation:")
        # Record into history manually for offline simulation
        hist = get_session_history("user-123")
        hist.add_user_message(q1)
        hist.add_ai_message(mock_answers[0])
        print(f"Assistant (offline): {mock_answers[0]}\n")

    # Turn 2 - Model has context of Turn 1
    print("--- Turn 2 (Testing Memory Context) ---", flush=True)
    q2 = "What are the main components I just asked about?"
    print(f"User: {q2}")
    try:
        r2 = chain_with_history.invoke({"input": q2}, config=config)
        print(f"Assistant: {r2}\n")
    except Exception as e:
        hist = get_session_history("user-123")
        hist.add_user_message(q2)
        hist.add_ai_message(mock_answers[1])
        print(f"Assistant (offline): {mock_answers[1]}\n")

    # 4. Inspecting Stored History
    print("--- 3. Inspecting In-Memory History Store ---", flush=True)
    history = get_session_history("user-123")
    print(f"Total messages in history for 'user-123': {len(history.messages)}")
    for i, msg in enumerate(history.messages, 1):
        print(f"  Turn {i} [{msg.type.upper()}]: {msg.content}")

    # 5. Session Isolation Check
    print("\n--- 4. Session Isolation Check (new user 'user-456') ---", flush=True)
    config_other = {"configurable": {"session_id": "user-456"}}
    hist_other = get_session_history("user-456")
    print(f"Messages in history for 'user-456': {len(hist_other.messages)} (clean state)")

    print("\n" + "=" * 60, flush=True)
    print("9.7 Conversational Memory Demo Completed Successfully!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_conversational_memory_demo()
