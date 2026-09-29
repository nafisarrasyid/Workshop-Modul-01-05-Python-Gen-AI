# Module 06 - OpenAI & Anthropic APIs
# 6.2 The Anthropic SDK - Multi-turn conversation
# The Anthropic API is stateless - you must send the full conversation history
# on every call. Build the history yourself.
#
# NOTE: This is an interactive CLI chat loop that makes REAL, billed API calls.
# Run it directly (python 04_anthropic_multiturn_chat.py) and type "exit" to quit.

import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OMNIROUTE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
base_url = os.getenv("OMNIROUTE_BASE_URL", "").rstrip("/")
if base_url.endswith("/v1"):
    base_url = base_url[:-3]

client = anthropic.Anthropic(
    api_key=api_key,
    base_url=base_url if base_url else None,
) if api_key else None
model = os.getenv("ANTHROPIC_MODEL", "antigravity/claude-sonnet-4-6")


def chat(system: str) -> None:
    """Simple interactive multi-turn chat loop."""
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        history.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=history,
        )

        assistant_text = response.content[0].text
        history.append({"role": "assistant", "content": assistant_text})
        print(f"Claude: {assistant_text}\n")


if __name__ == "__main__":
    if client is None:
        print("OMNIROUTE_API_KEY / ANTHROPIC_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        chat(system="You are a helpful Python tutor.")
