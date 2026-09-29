# Module 07 - Prompt Engineering
# 7.5 Structured Output (JSON Mode)
# Getting models to return valid, parseable JSON is one of the most common
# production requirements. There are two approaches.
#
# NOTE: Both demos make REAL, billed API calls (Anthropic and OpenAI
# respectively). They require ANTHROPIC_API_KEY / OPENAI_API_KEY in a .env
# file in this folder to run.

import json
import os

import anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ── Approach 1: Prompt-enforced JSON (works with any model) ──────────────────

ANTHROPIC_SYSTEM = """You are a data extractor. Extract information and return ONLY a JSON object.
No markdown, no explanation, no code fences. Raw JSON only.

Schema:
{
  "company": string,
  "founded": integer or null,
  "products": [string],
  "headquarters": string or null,
  "is_public": boolean
}"""

COMPANY_TEXTS = [
    "Anthropic was founded in 2021 by Dario Amodei and others. It makes Claude "
    "AI models and is headquartered in San Francisco. It is a private company.",
    "OpenAI, founded in 2015, created ChatGPT and GPT-4. Based in San "
    "Francisco, it remains private despite a major Microsoft investment.",
]


def extract_company_info(text: str) -> dict:
    api_key = os.getenv("OMNIROUTE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    base_url = os.getenv("OMNIROUTE_BASE_URL", "").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]

    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=base_url if base_url else None,
    )
    model = os.getenv("ANTHROPIC_MODEL", "antigravity/claude-sonnet-4-6")

    resp = client.messages.create(
        model=model,
        max_tokens=256,
        system=ANTHROPIC_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    raw = resp.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


def run_anthropic_demo() -> None:
    for text in COMPANY_TEXTS:
        info = extract_company_info(text)
        print(json.dumps(info, indent=2))
        print()


# ── Approach 2: OpenAI JSON mode (response_format enforces valid JSON) ───────

def run_openai_json_mode_demo() -> None:
    api_key = os.getenv("OMNIROUTE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.getenv("OMNIROUTE_BASE_URL", "").rstrip("/")
    if base_url and not base_url.endswith("/v1"):
        base_url += "/v1"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
    )
    model = os.getenv("OPENAI_MODEL", "antigravity/gpt-oss-120b-medium")

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": 'Extract entities. Return valid JSON: {"people": [string], "organizations": [string], "locations": [string]}',
                },
                {
                    "role": "user",
                    "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla.",
                },
            ],
        )
        content = response.choices[0].message.content
    except Exception:
        # Fallback for gateways/upstreams that do not support response_format parameter
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": 'Extract entities. Return valid JSON only with keys "people", "organizations", "locations". Output raw JSON without markdown or explanations.',
                },
                {
                    "role": "user",
                    "content": "Elon Musk founded SpaceX in Hawthorne, California. He also leads Tesla.",
                },
            ],
        )
        content = response.choices[0].message.content

    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    result = json.loads(content)
    print(result)
    # {'people': ['Elon Musk'], 'organizations': ['SpaceX', 'Tesla'], 'locations': ['Hawthorne, California']}


if __name__ == "__main__":
    if os.getenv("OMNIROUTE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"):
        print("=== Approach 1: Anthropic prompt-enforced JSON ===")
        run_anthropic_demo()
    else:
        print("Skipping Anthropic demo - OMNIROUTE_API_KEY / ANTHROPIC_API_KEY not set.")

    if os.getenv("OMNIROUTE_API_KEY") or os.getenv("OPENAI_API_KEY"):
        print("=== Approach 2: OpenAI json_object mode ===")
        run_openai_json_mode_demo()
    else:
        print("Skipping OpenAI demo - OMNIROUTE_API_KEY / OPENAI_API_KEY not set.")
