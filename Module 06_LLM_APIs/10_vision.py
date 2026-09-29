# Module 06 - OpenAI & Anthropic APIs
# 6.5 Vision - Images as Input
# Both APIs accept images alongside text in the same message.
#
# NOTE: This script makes REAL, billed API calls to Anthropic. It requires a
# valid ANTHROPIC_API_KEY in a .env file in this folder to run.

import anthropic
import os
import base64
from dotenv import load_dotenv

load_dotenv()

import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

api_key = os.getenv("OMNIROUTE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
base_url = os.getenv("OMNIROUTE_BASE_URL", "").rstrip("/")
if base_url.endswith("/v1"):
    base_url = base_url[:-3]

client = anthropic.Anthropic(
    api_key=api_key,
    base_url=base_url if base_url else None,
) if api_key else None
model = os.getenv("ANTHROPIC_MODEL", "antigravity/claude-sonnet-4-6")


# Option A: URL (download & convert to base64 if needed because Anthropic API requires base64 images)
def describe_image_url(url: str) -> str:
    import httpx
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = httpx.get(url, headers=headers, timeout=30.0)
    resp.raise_for_status()
    b64 = base64.standard_b64encode(resp.content).decode("utf-8")
    content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0]

    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": content_type,
                        "data": b64,
                    },
                },
                {"type": "text", "text": "Describe what you see in this image."},
            ],
        }],
    )
    return response.content[0].text


# Option B: base64 (for local files)
def describe_image_file(path: str) -> str:
    data = Path(path).read_bytes()
    b64 = base64.standard_b64encode(data).decode()
    ext = Path(path).suffix.lstrip(".").lower()
    media_type = f"image/{ext}"  # image/png, image/jpeg, image/webp, image/gif

    response = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                },
                {"type": "text", "text": "What is in this image?"},
            ],
        }],
    )
    return response.content[0].text


if __name__ == "__main__":
    if client is None:
        print("OMNIROUTE_API_KEY / ANTHROPIC_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        text = describe_image_url(
            "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/25.png"
        )
        print(text)
