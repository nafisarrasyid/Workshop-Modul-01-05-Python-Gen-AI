"""Module 14 - CrewAI
Section 14.3: Custom Tools for Agents

Demonstrates:
  - Subclassing crewai.tools.BaseTool for structured tools
  - Defining functional tools using the @tool decorator
  - Binding custom tools to CrewAI agents
"""
import sys
import json
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Type

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_chat_model

from crewai import Agent
from crewai.tools import BaseTool, tool


# 1. Class-based BaseTool
class BenchmarkSearchArgs(BaseModel):
    query: str = Field(..., description="The query to search in the benchmark database.")

class ModelBenchmarkTool(BaseTool):
    name: str = "ModelBenchmarkSearch"
    description: str = "Searches latency, throughput, and accuracy scores for GenAI models."
    args_schema: Type[BaseModel] = BenchmarkSearchArgs

    def _run(self, query: str = "", **kwargs) -> str:
        benchmarks = {
            "gpt-4o": {"latency_ms": 320, "throughput_tok_s": 105, "eval_mmlu": 88.7},
            "claude-sonnet-4-5": {"latency_ms": 380, "throughput_tok_s": 92, "eval_mmlu": 89.2},
            "gemini-2-flash": {"latency_ms": 190, "throughput_tok_s": 160, "eval_mmlu": 86.4},
        }
        for name, metrics in benchmarks.items():
            if name in query.lower():
                return f"Benchmark for {name}: Latency={metrics['latency_ms']}ms, Throughput={metrics['throughput_tok_s']} t/s, MMLU={metrics['eval_mmlu']}%"
        return f"Benchmark search for '{query}': Average industry latency 300ms, throughput 110 t/s."


# 2. Decorator-based Tool
@tool(name="CostEstimator", description="Estimates monthly LLM API operational costs.")
def estimate_cost(monthly_tokens: int = 10_000_000, price_per_million: float = 2.5) -> str:
    total = (monthly_tokens / 1_000_000) * price_per_million
    return f"Estimated monthly cost for {monthly_tokens:,} tokens at ${price_per_million}/M tokens is ${total:.2f} USD."


def main():
    print("=" * 60)
    print("Module 14.3: Custom Tools for CrewAI Agents")
    print("=" * 60)

    # 3. Instantiate and verify tools directly
    b_tool = ModelBenchmarkTool()
    print("Executing ModelBenchmarkTool:")
    res_b = b_tool.run(query="gpt-4o benchmark stats")
    print(f"  Result: {res_b}")

    print("\nExecuting CostEstimator Tool:")
    res_c = estimate_cost.run(monthly_tokens=25_000_000, price_per_million=2.5)
    print(f"  Result: {res_c}")

    # 4. Attach tools to Agent
    researcher = Agent(
        role="Systems Benchmark Specialist",
        goal="Analyze model latency and compute costs for system design",
        backstory="An engineer focused on optimizing model inference efficiency.",
        tools=[b_tool, estimate_cost],
        llm=get_chat_model(),
        verbose=True,
    )

    print(f"\nAgent '{researcher.role}' initialized with tools:")
    for t in researcher.tools:
        print(f"  - {t.name}: {t.description}")

    print("\nTools defined and wired successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
