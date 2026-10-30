"""Module 14 - CrewAI
Section 14.4: Tasks & Context Dependencies

Demonstrates:
  - Creating Task instances with clear descriptions and expected outputs
  - Assigning Tasks to designated specialist Agents
  - Wiring Task dependencies using the `context` parameter
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_chat_model

from crewai import Agent, Task

def main():
    print("=" * 60)
    print("Module 14.4: Tasks & Context Dependencies")
    print("=" * 60)

    llm = get_chat_model()

    # Define specialist agents
    researcher = Agent(
        role="AI Architecture Researcher",
        goal="Identify key trade-offs in modern vector database design",
        backstory="An expert in distributed index structures and HNSW graphs.",
        llm=llm,
    )
    writer = Agent(
        role="Technical Editor",
        goal="Draft an executive summary based strictly on research findings",
        backstory="A clear writer specializing in translating research into executive briefs.",
        llm=llm,
    )

    # 1. Primary Research Task
    research_task = Task(
        description="Analyze trade-offs between ChromaDB, Pinecone, and Weaviate for production RAG.",
        expected_output="A structured list of 3 pros and 3 cons for each vector database.",
        agent=researcher,
    )

    # 2. Dependent Writing Task (receives research_task output via `context`)
    writing_task = Task(
        description="Draft a 2-paragraph executive recommendation based on the research findings.",
        expected_output="A polished executive brief recommending the optimal vector DB for enterprise use.",
        agent=writer,
        context=[research_task],  # Context dependency!
    )

    print("Defined Tasks:")
    print(f"Task 1: '{research_task.description}'")
    print(f"  Assigned Agent : {research_task.agent.role}")
    print(f"  Expected Output: {research_task.expected_output}")

    print(f"\nTask 2: '{writing_task.description}'")
    print(f"  Assigned Agent : {writing_task.agent.role}")
    print(f"  Context Source : {[t.description[:35] + '...' for t in writing_task.context]}")
    print(f"  Expected Output: {writing_task.expected_output}")

    print("\nTask dependencies wired successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
