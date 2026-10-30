"""Module 14 - CrewAI
Section 14.5: Sequential Crew Execution

Demonstrates:
  - Assembling Agents and Tasks into a Crew
  - Executing end-to-end multi-agent pipelines with Process.sequential
  - Passing outputs automatically between dependent tasks
  - Inspecting individual task outputs and the final crew deliverable
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_chat_model

from crewai import Agent, Task, Crew, Process

def main():
    print("=" * 60)
    print("Module 14.5: Sequential Crew Pipeline")
    print("=" * 60)

    llm = get_chat_model()

    # 1. Define Agents
    analyst = Agent(
        role="Senior Technology Analyst",
        goal="Analyze enterprise adoption trends of agentic workflows",
        backstory="A market researcher analyzing AI deployment patterns.",
        llm=llm,
        verbose=True,
    )
    writer = Agent(
        role="Content Specialist",
        goal="Produce an informative report from research findings",
        backstory="A technical communicator adept at clean markdown reports.",
        llm=llm,
        verbose=True,
    )

    # 2. Define Tasks with Sequential Context
    task_research = Task(
        description="Identify the top 3 drivers for multi-agent system adoption in 2026.",
        expected_output="3 key drivers with brief bulleted explanations.",
        agent=analyst,
    )
    task_write = Task(
        description="Format the 3 drivers into an executive summary with a concluding recommendation.",
        expected_output="A markdown summary containing key drivers and next steps.",
        agent=writer,
        context=[task_research],
    )

    # 3. Create Crew with Process.sequential
    tech_crew = Crew(
        agents=[analyst, writer],
        tasks=[task_research, task_write],
        process=Process.sequential,
        verbose=True,
    )

    print("\nStarting Sequential Crew Kickoff...\n")
    crew_result = tech_crew.kickoff()

    print("\n" + "=" * 60)
    print("Crew Kickoff Finished. Final Deliverable:")
    print("=" * 60)
    print(str(crew_result))

    print("\nSequential Crew executed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
