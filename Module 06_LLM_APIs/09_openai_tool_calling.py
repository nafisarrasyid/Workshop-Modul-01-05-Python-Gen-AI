# Module 06 - OpenAI & Anthropic APIs
# 6.4 Tool Calling (Function Calling) - OpenAI
#
# NOTE: This script makes REAL, billed API calls to OpenAI (two calls per
# run). It requires a valid OPENAI_API_KEY in a .env file in this folder.

from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_model_info",
            "description": "Returns context window and pricing for a given LLM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string", "description": "Model identifier."}
                },
                "required": ["model_name"],
            },
        },
    }
]


def get_model_info(model_name: str) -> dict:
    db = {
        "gpt-4o":                          {"context_k": 128, "cost_input": 2.50},
        "antigravity/gpt-oss-120b-medium": {"context_k": 128, "cost_input": 0.00},
        "claude-sonnet-4-5":               {"context_k": 200, "cost_input": 3.00},
        "antigravity/claude-sonnet-4-6":   {"context_k": 200, "cost_input": 3.00},
    }
    return db.get(model_name, {"error": "unknown model"})


def run_tool_call_demo() -> None:
    api_key = os.getenv("OMNIROUTE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.getenv("OMNIROUTE_BASE_URL", "").rstrip("/")
    if base_url and not base_url.endswith("/v1"):
        base_url += "/v1"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
    )
    model = os.getenv("OPENAI_MODEL", "antigravity/gpt-oss-120b-medium")

    messages = [{"role": "user", "content": f"What is {model}'s context window?"}]

    response = client.chat.completions.create(
        model=model,
        tools=TOOLS,
        messages=messages,
    )

    if response.choices[0].finish_reason == "tool_calls":
        tool_call = response.choices[0].message.tool_calls[0]
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        result = get_model_info(**args)

        # Append assistant message and tool result
        messages.append(response.choices[0].message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        })

        final = client.chat.completions.create(model=model, messages=messages)
        print(final.choices[0].message.content)


if __name__ == "__main__":
    if not (os.getenv("OMNIROUTE_API_KEY") or os.getenv("OPENAI_API_KEY")):
        print("OMNIROUTE_API_KEY / OPENAI_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        run_tool_call_demo()
