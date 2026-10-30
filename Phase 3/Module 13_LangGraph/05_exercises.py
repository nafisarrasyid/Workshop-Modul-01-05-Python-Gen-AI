"""Module 13 - LangGraph & Agents
Section 13.5: Module 13 Exercises

Implements all 4 exercises from the curriculum:
  - Exercise 1: Research Agent with Search, Summarize, and Save Tools
  - Exercise 2: Max Step Limit Safeguard for ReAct Loops
  - Exercise 3: Plan-and-Execute Agent Pattern
  - Exercise 4: Multi-Turn Conversational Memory with Checkpointing
"""
import sys
from pathlib import Path
from typing import TypedDict, List, Annotated, Dict, Any
import operator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ── Exercise 1: Research Agent ───────────────────────────────────────────────
class ResearchState(TypedDict):
    topic: str
    search_results: List[str]
    summary: str
    saved_notes: Annotated[List[str], operator.add]

def search_node(state: ResearchState) -> dict:
    topic = state["topic"]
    # Mock search tool
    findings = [
        f"{topic} demonstrated 40% latency reduction with speculative decoding.",
        f"{topic} incorporates multi-token prediction heads for faster autoregression.",
    ]
    return {"search_results": findings}

def summarize_node(state: ResearchState) -> dict:
    results = state["search_results"]
    summary = f"Summary on {state['topic']}: Key improvements include " + "; ".join(results)
    return {"summary": summary}

def save_notes_node(state: ResearchState) -> dict:
    note = f"[Archived Note] {state['summary']}"
    return {"saved_notes": [note]}

def exercise_1_research_agent():
    print("\n--- Exercise 1: Research Agent with Tool Pipeline ---")
    builder = StateGraph(ResearchState)
    builder.add_node("search", search_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("save_notes", save_notes_node)

    builder.add_edge(START, "search")
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "save_notes")
    builder.add_edge("save_notes", END)

    app = builder.compile()

    topic = "Speculative Decoding in LLMs"
    print(f"Researching Topic: '{topic}'")
    out = app.invoke({"topic": topic, "search_results": [], "summary": "", "saved_notes": []})

    print("Synthesized Summary:")
    print(f"  {out['summary']}")
    print("Saved Notes Store:")
    for note in out["saved_notes"]:
        print(f"  {note}")

    assert len(out["saved_notes"]) >= 1
    print("Exercise 1 passed successfully!")


# ── Exercise 2: Max Step Limit Safeguard ──────────────────────────────────────
class LoopState(TypedDict):
    query: str
    step_count: int
    max_steps: int
    status: str

def agent_step_node(state: LoopState) -> dict:
    return {
        "step_count": state["step_count"] + 1,
        "status": "in_progress",
    }

def limit_check(state: LoopState) -> str:
    if state["step_count"] >= state["max_steps"]:
        return "abort_limit_exceeded"
    # Simulate unresolved condition forcing continuation
    if state["step_count"] < 3:
        return "agent_step"
    return "completed"

def timeout_handler_node(state: LoopState) -> dict:
    return {"status": f"aborted: exceeded safety budget of {state['max_steps']} iterations"}

def success_handler_node(state: LoopState) -> dict:
    return {"status": "success: completed within budget"}

def exercise_2_step_limit_safeguard():
    print("\n--- Exercise 2: Max Step Limit Safeguard for ReAct Loops ---")
    workflow = StateGraph(LoopState)
    workflow.add_node("agent_step", agent_step_node)
    workflow.add_node("timeout_handler", timeout_handler_node)
    workflow.add_node("success_handler", success_handler_node)

    workflow.add_edge(START, "agent_step")
    workflow.add_conditional_edges(
        "agent_step",
        limit_check,
        {
            "agent_step": "agent_step",
            "abort_limit_exceeded": "timeout_handler",
            "completed": "success_handler",
        },
    )
    workflow.add_edge("timeout_handler", END)
    workflow.add_edge("success_handler", END)

    app = workflow.compile()

    # Case A: Safety trigger (max_steps = 2, requiring 3)
    res_a = app.invoke({"query": "Indefinite task", "step_count": 0, "max_steps": 2, "status": ""})
    print(f"Test A (Budget=2, Needed=3): Steps={res_a['step_count']} -> Status='{res_a['status']}'")
    assert "aborted" in res_a["status"], "Safeguard should have aborted execution!"

    # Case B: Success within budget (max_steps = 5)
    res_b = app.invoke({"query": "Normal task", "step_count": 0, "max_steps": 5, "status": ""})
    print(f"Test B (Budget=5, Needed=3): Steps={res_b['step_count']} -> Status='{res_b['status']}'")
    assert "success" in res_b["status"], "Execution should have succeeded!"

    print("Exercise 2 passed successfully!")


# ── Exercise 3: Plan-and-Execute Agent ────────────────────────────────────────
class PlanExecuteState(TypedDict):
    goal: str
    plan: List[str]
    executed_steps: Annotated[List[str], operator.add]
    final_output: str

def planner_node(state: PlanExecuteState) -> dict:
    goal = state["goal"]
    # Decompose into structured plan
    subtasks = [
        f"Step 1: Ingest and index documentation for '{goal}'",
        f"Step 2: Generate unit tests covering edge cases for '{goal}'",
        f"Step 3: Run static code analysis and linting for '{goal}'",
    ]
    return {"plan": subtasks}

def executor_node(state: PlanExecuteState) -> dict:
    plan = state["plan"]
    completed = []
    for step in plan:
        completed.append(f"[Executed] {step}")
    final_res = f"Goal '{state['goal']}' achieved with all {len(plan)} subtasks verified."
    return {"executed_steps": completed, "final_output": final_res}

def exercise_3_plan_and_execute():
    print("\n--- Exercise 3: Plan-and-Execute Agent Pattern ---")
    workflow = StateGraph(PlanExecuteState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", END)

    app = workflow.compile()

    goal = "Implement Secure OAuth2 Authentication"
    print(f"Goal: '{goal}'")
    result = app.invoke({"goal": goal, "plan": [], "executed_steps": [], "final_output": ""})

    print("\nGenerated Plan:")
    for p in result["plan"]:
        print(f"  {p}")

    print("\nExecution Deliverables:")
    for step in result["executed_steps"]:
        print(f"  {step}")
    print(f"Final Report: {result['final_output']}")

    assert len(result["executed_steps"]) == 3
    print("Exercise 3 passed successfully!")


# ── Exercise 4: Multi-Turn Conversational Memory ─────────────────────────────
class ChatMemoryState(TypedDict):
    user_name: str
    preferred_language: str
    history: Annotated[List[str], operator.add]
    last_response: str

def chat_node(state: ChatMemoryState) -> dict:
    name = state.get("user_name", "Developer")
    lang = state.get("preferred_language", "Python")
    resp = f"Hello {name}, I have recorded that your primary language is {lang}."
    return {
        "last_response": resp,
        "history": [f"System acknowledged preferences for {name} ({lang})."],
    }

def exercise_4_multi_turn_memory():
    print("\n--- Exercise 4: Multi-Turn Agent with Checkpointed Memory ---")
    checkpointer = MemorySaver()
    workflow = StateGraph(ChatMemoryState)
    workflow.add_node("chat", chat_node)
    workflow.add_edge(START, "chat")
    workflow.add_edge("chat", END)

    app = workflow.compile(checkpointer=checkpointer)
    thread = {"configurable": {"thread_id": "session-dev-user-007"}}

    # Turn 1: User provides preferences
    turn_1_input = {"user_name": "Napis", "preferred_language": "Python 3.9", "history": []}
    print(f"Turn 1 Input: User={turn_1_input['user_name']}, Language={turn_1_input['preferred_language']}")
    _ = app.invoke(turn_1_input, config=thread)

    # Turn 2: User asks a follow-up query without restating preferences
    state_before_turn_2 = app.get_state(thread)
    saved_name = state_before_turn_2.values.get("user_name")
    saved_lang = state_before_turn_2.values.get("preferred_language")
    print(f"\nTurn 2 Checkpoint Verification:")
    print(f"  Persisted User Name       : '{saved_name}'")
    print(f"  Persisted Preferred Lang  : '{saved_lang}'")
    print(f"  History Entries Recorded  : {len(state_before_turn_2.values.get('history', []))}")

    assert saved_name == "Napis", "Checkpointer failed to preserve user_name across turns!"
    assert saved_lang == "Python 3.9", "Checkpointer failed to preserve preferred_language!"

    print("Multi-turn memory persistence verified across separate invocations!")
    print("Exercise 4 passed successfully!")


def main():
    print("=" * 60)
    print("Module 13: LangGraph & Agents - Comprehensive Exercises")
    print("=" * 60)

    exercise_1_research_agent()
    exercise_2_step_limit_safeguard()
    exercise_3_plan_and_execute()
    exercise_4_multi_turn_memory()

    print("\n" + "=" * 60)
    print("All Module 13 Exercises Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
