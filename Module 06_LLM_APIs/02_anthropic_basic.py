# Module 06 - OpenAI & Anthropic APIs
# 6.2 The Anthropic SDK - Basic message call
#
# NOTE: This script makes a REAL, billed API call to Anthropic. It requires a
# valid ANTHROPIC_API_KEY in a .env file in this folder to run.

import anthropic
import os
from dotenv import load_dotenv

load_dotenv()


def basic_call() -> None:
    api_key = os.getenv("OMNIROUTE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.getenv("OMNIROUTE_BASE_URL", "").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]  # Anthropic SDK handles /v1 automatically

    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=base_url if base_url else None,
    )
    model = os.getenv("ANTHROPIC_MODEL", "antigravity/claude-sonnet-4-6")

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What is retrieval-augmented generation?"}
        ],
    )

    # Response text is in the first content block
    print(message.content[0].text)

    # Usage stats
    print(f"Input tokens: {message.usage.input_tokens}")
    print(f"Output tokens: {message.usage.output_tokens}")


if __name__ == "__main__":
    if not (os.getenv("OMNIROUTE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("OMNIROUTE_API_KEY / ANTHROPIC_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        basic_call()
