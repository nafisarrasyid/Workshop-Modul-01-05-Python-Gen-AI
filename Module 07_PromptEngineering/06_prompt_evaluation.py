# Module 07 - Prompt Engineering
# 7.7 Prompt Evaluation
# A prompt that works once is not a prompt - it is luck. Systematically
# evaluate prompts across a test set before using them in production.
#
# NOTE: evaluate_prompt() makes one REAL, billed API call per test case. It
# requires a valid ANTHROPIC_API_KEY in a .env file in this folder to run.

import anthropic
import os
import json
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EvalCase:
    input_text: str
    expected_keywords: list[str]   # at least one must appear in response
    must_be_json: bool = False


def evaluate_prompt(system: str, cases: list[EvalCase], client=None) -> dict:
    """Run a prompt against test cases and return pass rate + details."""
    if client is None:
        api_key = os.getenv("OMNIROUTE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        base_url = os.getenv("OMNIROUTE_BASE_URL", "").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url if base_url else None,
        )
    model = os.getenv("ANTHROPIC_MODEL", "antigravity/claude-sonnet-4-6")
    results = []

    for case in cases:
        resp = client.messages.create(
            model=model,
            max_tokens=256,
            system=system,
            messages=[{"role": "user", "content": case.input_text}],
        )
        text = resp.content[0].text.strip()

        # Check keyword hit
        keyword_hit = any(kw.lower() in text.lower() for kw in case.expected_keywords)

        # Check JSON validity if required
        json_valid = True
        if case.must_be_json:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                json_valid = False

        passed = keyword_hit and json_valid
        results.append({
            "input": case.input_text[:60],
            "passed": passed,
            "response_preview": text[:80],
        })

    pass_rate = sum(r["passed"] for r in results) / len(results)
    return {"pass_rate": pass_rate, "results": results}


# Test a classification prompt
CLASSIFY_SYSTEM = """Classify the AI task as one of: CLASSIFICATION, GENERATION, RETRIEVAL, EMBEDDING.
Return ONLY the category word."""

TEST_CASES = [
    EvalCase("Predict whether an email is spam.", ["CLASSIFICATION"]),
    EvalCase("Write a product description for headphones.", ["GENERATION"]),
    EvalCase("Find the most relevant documents for a query.", ["RETRIEVAL"]),
    EvalCase("Convert this sentence to a vector.", ["EMBEDDING"]),
    EvalCase("Label customer reviews as positive or negative.", ["CLASSIFICATION"]),
]


if __name__ == "__main__":
    if not (os.getenv("OMNIROUTE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("OMNIROUTE_API_KEY / ANTHROPIC_API_KEY is not set. Add it to a .env file in this folder.")
    else:
        report = evaluate_prompt(CLASSIFY_SYSTEM, TEST_CASES)
        print(f"Pass rate: {report['pass_rate']:.0%}")
        for r in report["results"]:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] {r['input']!r} -> {r['response_preview']!r}")
