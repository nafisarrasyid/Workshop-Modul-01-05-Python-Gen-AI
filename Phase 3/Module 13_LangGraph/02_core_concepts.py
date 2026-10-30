"""Module 13 - LangGraph & Agents
Section 13.2: Core Concepts

Demonstrates:
  - TypedDict State definition
  - Node functions that read and update graph state
  - Adding nodes, normal edges, and conditional edges
  - Compiling and executing a StateGraph from START to END
"""
import sys
from pathlib import Path
from typing import TypedDict, Annotated, List
import operator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, START, END

# 1. Define State schema using TypedDict
class WorkflowState(TypedDict):
    input_text: str
    tokens_count: int
    sentiment: str
    action_log: Annotated[List[str], operator.add]


# 2. Node Functions
def analyze_node(state: WorkflowState) -> dict:
    text = state["input_text"]
    words = text.split()
    # Simple rule-based sentiment
    pos_words = {"great", "excellent", "fast", "innovative", "seamless"}
    neg_words = {"slow", "error", "broken", "bug", "crash"}
    
    score = sum(1 for w in words if w.lower() in pos_words) - sum(1 for w in words if w.lower() in neg_words)
    sentiment = "positive" if score >= 0 else "negative"

    return {
        "tokens_count": len(words),
        "sentiment": sentiment,
        "action_log": [f"Analyzed text: {len(words)} words, sentiment={sentiment}"],
    }


def handle_positive_node(state: WorkflowState) -> dict:
    return {
        "action_log": ["Executed positive routing: Forwarded to engagement team."]
    }


def handle_negative_node(state: WorkflowState) -> dict:
    return {
        "action_log": ["Executed negative routing: Created high-priority incident ticket."]
    }


# 3. Conditional Edge Routing function
def route_by_sentiment(state: WorkflowState) -> str:
    if state["sentiment"] == "positive":
        return "positive_path"
    return "negative_path"


def main():
    print("=" * 60)
    print("Module 13.2: LangGraph Core Concepts & StateGraph")
    print("=" * 60)

    # 4. Construct Graph
    builder = StateGraph(WorkflowState)

    builder.add_node("analyze", analyze_node)
    builder.add_node("positive_path", handle_positive_node)
    builder.add_node("negative_path", handle_negative_node)

    builder.add_edge(START, "analyze")
    builder.add_conditional_edges(
        "analyze",
        route_by_sentiment,
        {
            "positive_path": "positive_path",
            "negative_path": "negative_path",
        },
    )
    builder.add_edge("positive_path", END)
    builder.add_edge("negative_path", END)

    app = builder.compile()

    # 5. Run test 1: Positive input
    sample_1 = {"input_text": "Antigravity provides seamless, innovative and fast agent workflows.", "action_log": []}
    print(f"Running Input 1: '{sample_1['input_text']}'")
    out_1 = app.invoke(sample_1)
    print(f"Final Sentiment: {out_1['sentiment']}")
    print("Action Log:")
    for log in out_1["action_log"]:
        print(f"  - {log}")

    # Run test 2: Negative input
    sample_2 = {"input_text": "Encountered a slow crash and build error during deployment.", "action_log": []}
    print(f"\nRunning Input 2: '{sample_2['input_text']}'")
    out_2 = app.invoke(sample_2)
    print(f"Final Sentiment: {out_2['sentiment']}")
    print("Action Log:")
    for log in out_2["action_log"]:
        print(f"  - {log}")

    print("\nStateGraph compiled and executed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
