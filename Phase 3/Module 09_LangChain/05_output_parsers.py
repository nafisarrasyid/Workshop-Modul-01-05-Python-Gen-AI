# Module 09 - LangChain
# 9.5 Output Parsers
#
# Output parsers convert raw LLM text/tool outputs into structured Python types.
# Covered in this file:
#   1. Structured Output with Pydantic via `llm.with_structured_output()`
#   2. JsonOutputParser with format instructions for portable structured extraction
#   3. StrOutputParser for clean string streaming and extraction

import json
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# Import utils first to apply compatibility patches
from utils import get_chat_model

from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.fake_chat_models import FakeListChatModel


# ── Pydantic Schema for Structured Output ────────────────────────────────────
class ModelComparison(BaseModel):
    models: List[str]     = Field(description="Model names compared")
    best_for_cost: str    = Field(description="Most cost-effective model")
    best_for_quality: str = Field(description="Highest quality model")
    recommendation: str   = Field(description="Overall recommendation")


def run_output_parsers_demo() -> None:
    print("=" * 60, flush=True)
    print("9.5 Output Parsers - Structured Outputs & Pydantic Validation", flush=True)
    print("=" * 60, flush=True)

    models_input = "gpt-4o, gpt-4o-mini, claude-sonnet-4-5"

    # Simulated fallback output
    sample_structured = ModelComparison(
        models=["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-5"],
        best_for_cost="gpt-4o-mini",
        best_for_quality="claude-sonnet-4-5",
        recommendation="Use gpt-4o-mini for routing and bulk preprocessing, and claude-sonnet-4-5 for complex reasoning.",
    )

    # ── 1. Structured Output with with_structured_output ─────────────────────
    print("\n--- 1. Schema-Enforced Output (with_structured_output) ---", flush=True)
    llm = get_chat_model()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI model comparison expert."),
        ("human",  "Compare these models for a startup on a budget: {models}"),
    ])

    executed_online = False
    if hasattr(llm, "with_structured_output"):
        try:
            structured_llm = llm.with_structured_output(ModelComparison)
            chain = prompt | structured_llm
            result: ModelComparison = chain.invoke({"models": models_input})
            print(f"Models Compared: {', '.join(result.models)}")
            print(f"Best for cost:   {result.best_for_cost}")
            print(f"Best for quality:{result.best_for_quality}")
            print(f"Recommendation:  {result.recommendation}")
            executed_online = True
        except Exception as e:
            print(f"[API unreachable / provider note: {e}] Running offline simulation:")

    if not executed_online:
        print(f"Models Compared: {', '.join(sample_structured.models)}")
        print(f"Best for cost:   {sample_structured.best_for_cost}")
        print(f"Best for quality:{sample_structured.best_for_quality}")
        print(f"Recommendation:  {sample_structured.recommendation}")

    # ── 2. JsonOutputParser with Format Instructions ─────────────────────────
    print("\n--- 2. Portable JsonOutputParser ---", flush=True)
    json_parser = JsonOutputParser(pydantic_object=ModelComparison)
    format_instructions = json_parser.get_format_instructions()

    json_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI model comparison expert.\n{format_instructions}"),
        ("human",  "Compare these models for a startup on a budget: {models}"),
    ]).partial(format_instructions=format_instructions)

    mock_json_str = json.dumps(sample_structured.model_dump())
    json_chain = json_prompt | FakeListChatModel(responses=[mock_json_str]) | json_parser
    parsed_json = json_chain.invoke({"models": models_input})

    print("Parsed JSON dict:")
    for k, v in parsed_json.items():
        print(f"  {k}: {v}")

    # ── 3. StrOutputParser ───────────────────────────────────────────────────
    print("\n--- 3. StrOutputParser ---", flush=True)
    str_chain = ChatPromptTemplate.from_template("Summarize {models} in 1 line.") | FakeListChatModel(responses=["Fast, efficient, and versatile frontier model suite."]) | StrOutputParser()
    print("Raw string output:", str_chain.invoke({"models": models_input}))

    print("\n" + "=" * 60, flush=True)
    print("9.5 Output Parsers Demo Completed Successfully!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_output_parsers_demo()
