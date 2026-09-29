# Module 06 - OpenAI & Anthropic APIs
# 6.3 The OpenAI SDK - Streaming with OpenAI
#
# NOTE: This script makes a REAL, billed API call to OpenAI. It requires a
# valid OPENAI_API_KEY in a .env file in this folder to run.

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def stream_response() -> None:
    api_key = os.getenv("OMNIROUTE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.getenv("OMNIROUTE_BASE_URL", "").rstrip("/")
    if base_url and not base_url.endswith("/v1"):
        base_url += "/v1"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
    )
    model = os.getenv("OPENAI_MODEL", "antigravity/gpt-oss-120b-medium")

    stream = client.chat.completions.create(
        model=model,
        max_tokens=512,
        stream=True,
        messages=[{"role": "user", "content": "Explain embeddings in 3 bullet points."}],
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)

    print()


if __name__ == "__main__":
    if not (os.getenv("OMNIROUTE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("OMNIROUTE_API_KEY / OPENAI_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        stream_response()
