"""Module 13 - LangGraph & Agents
Section 13.3: ReAct Agent with Tool Calling Loop

Demonstrates:
  - Building a ReAct (Reasoning + Acting) loop using LangGraph
  - Defining executable tools with @tool
  - State management with add_messages reducer
  - Conditional loop: model -> should_continue -> tools -> model -> END
"""
import sys
import json
from pathlib import Path
from typing import TypedDict, Annotated, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_chat_model

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# 1. Define Tools
@tool
def get_weather(city: str) -> str:
    """Returns the current weather conditions for a given city."""
    weather_data = {
        "jakarta": "Sunny, 31C, 75% humidity",
        "tokyo": "Clear, 18C, 50% humidity",
        "london": "Rainy, 12C, 90% humidity",
    }
    return weather_data.get(city.lower(), f"Weather data unavailable for {city}.")

@tool
def calculate_trip_budget(days: int, daily_rate: float) -> str:
    """Calculates total accommodation budget for a trip."""
    total = days * daily_rate
    return f"Total estimated budget for {days} days at ${daily_rate}/day is ${total:.2f}."

tools = [get_weather, calculate_trip_budget]
tools_by_name = {t.name: t for t in tools}


# 2. Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# 3. Agent & Tool Nodes
def call_model(state: AgentState) -> dict:
    messages = state["messages"]
    last_msg = messages[-1]

    # Rule-based / simulated LLM reasoning for deterministic local verification
    if isinstance(last_msg, HumanMessage):
        query = last_msg.content.lower()
        if "weather in tokyo" in query:
            # Generate AIMessage with tool call
            ai_msg = AIMessage(
                content="I need to check the current weather in Tokyo first.",
                tool_calls=[{
                    "name": "get_weather",
                    "args": {"city": "Tokyo"},
                    "id": "call_weather_tokyo_01",
                }],
            )
            return {"messages": [ai_msg]}

    elif isinstance(last_msg, ToolMessage):
        # Observation received -> synthesize final answer
        final_answer = AIMessage(
            content=f"Based on the live forecast, the weather in Tokyo is currently {last_msg.content}."
        )
        return {"messages": [final_answer]}

    return {"messages": [AIMessage(content="I can assist you with weather forecasts and trip budgeting.")]}


def call_tools(state: AgentState) -> dict:
    messages = state["messages"]
    last_message = messages[-1]
    tool_messages = []

    for tool_call in getattr(last_message, "tool_calls", []):
        t_name = tool_call["name"]
        t_args = tool_call["args"]
        t_id = tool_call["id"]

        if t_name in tools_by_name:
            res = tools_by_name[t_name].invoke(t_args)
        else:
            res = f"Error: Tool '{t_name}' not found."

        tool_messages.append(ToolMessage(content=str(res), tool_call_id=t_id))

    return {"messages": tool_messages}


# 4. Conditional Edge Function
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def main():
    print("=" * 60)
    print("Module 13.3: ReAct Agent with Tool Execution Loop")
    print("=" * 60)

    # 5. Build Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent")

    app = workflow.compile()

    # 6. Run ReAct Query
    query = "What is the current weather in Tokyo?"
    print(f"User Query: '{query}'\n")

    initial_state = {"messages": [HumanMessage(content=query)]}
    final_output = app.invoke(initial_state)

    print("Execution History:")
    for idx, msg in enumerate(final_output["messages"], start=1):
        m_type = msg.__class__.__name__
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  [{idx}] {m_type}: {msg.content}")
            print(f"       -> Invoking Tool: {msg.tool_calls[0]['name']}({msg.tool_calls[0]['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"  [{idx}] ToolMessage (Observation): {msg.content}")
        else:
            print(f"  [{idx}] {m_type}: {msg.content}")

    print("\nReAct Agent completed loop successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
