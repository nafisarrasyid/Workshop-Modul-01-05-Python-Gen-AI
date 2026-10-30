"""Module 14 - CrewAI
Section 14.2: Agents

Demonstrates:
  - Defining specialized agents with roles, goals, and backstories
  - Configuring delegation permissions (allow_delegation)
  - Integrating LangChain ChatModels or shared robust resolvers
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_chat_model

from crewai import Agent

def main():
    print("=" * 60)
    print("Module 14.2: CrewAI Agent Definitions")
    print("=" * 60)

    llm = get_chat_model()

    # 1. Researcher Agent
    researcher = Agent(
        role="Senior AI Research Analyst",
        goal="Discover and synthesize cutting-edge advancements in agentic AI frameworks",
        backstory=(
            "You are a world-renowned AI researcher who has authored foundational papers "
            "on autonomous agent architectures and multi-agent coordination."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # 2. Technical Writer Agent
    writer = Agent(
        role="Lead Technical Writer",
        goal="Transform complex technical findings into clear, compelling engineering guides",
        backstory=(
            "You are an experienced technical author who excels at creating structured, "
            "production-grade documentation for enterprise engineering teams."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    # 3. Reviewer Agent
    reviewer = Agent(
        role="Principal Quality & Security Reviewer",
        goal="Verify technical correctness, architecture soundness, and security standards",
        backstory=(
            "You are a rigorous systems architect with 15 years of experience reviewing "
            "mission-critical software designs for reliability and vulnerability prevention."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )

    print("Created CrewAI Agents:")
    for a in [researcher, writer, reviewer]:
        print(f"\n[Role]: {a.role}")
        print(f"  Goal       : {a.goal}")
        print(f"  Backstory  : {a.backstory[:80]}...")
        print(f"  Delegation : {a.allow_delegation}")

    print("\nAgents configured successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
