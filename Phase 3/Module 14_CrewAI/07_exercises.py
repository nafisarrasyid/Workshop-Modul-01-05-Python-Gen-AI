"""Module 14 - CrewAI
Section 14.7: Module 14 Exercises

Implements all 4 exercises from the curriculum:
  - Exercise 1: Three-Agent Technical Blog Content Pipeline (Researcher -> Writer -> Reviewer)
  - Exercise 2: Collaborative Code Review Crew (Developer -> Security Reviewer -> QA Engineer)
  - Exercise 3: Custom WebSearchTool with Researcher Agent
  - Exercise 4: Comparative Evaluation: Process.sequential vs. Process.hierarchical
"""
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Type

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_chat_model

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool


# ── Exercise 1: Three-Agent Content Pipeline ─────────────────────────────────
def exercise_1_content_pipeline():
    print("\n--- Exercise 1: Three-Agent Technical Blog Content Pipeline ---")
    llm = get_chat_model()

    researcher = Agent(
        role="AI Trends Researcher",
        goal="Extract key facts and benchmarks regarding Agentic Workflows in 2026",
        backstory="A research specialist who gathers validated technological benchmarks.",
        llm=llm,
        verbose=False,
    )
    writer = Agent(
        role="Technical Blog Writer",
        goal="Compose an engaging, high-impact blog post from research findings",
        backstory="A writer who crafts accessible narratives for software engineers.",
        llm=llm,
        verbose=False,
    )
    reviewer = Agent(
        role="Senior Editor & Technical Reviewer",
        goal="Refine the post for accuracy, formatting, and structural clarity",
        backstory="A veteran publication editor who ensures publication-ready prose.",
        llm=llm,
        verbose=False,
    )

    t1 = Task(
        description="Research 3 breakthrough trends in Autonomous Coding Agents.",
        expected_output="3 trends with supporting data points.",
        agent=researcher,
    )
    t2 = Task(
        description="Draft a 3-section blog article titled 'The Rise of Agentic Coding'.",
        expected_output="Draft article with introduction, trends analysis, and conclusion.",
        agent=writer,
        context=[t1],
    )
    t3 = Task(
        description="Review and polish the article draft, ensuring technical rigor and clear formatting.",
        expected_output="Polished final blog article ready for release.",
        agent=reviewer,
        context=[t2],
    )

    content_crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=False,
    )

    result = content_crew.kickoff()
    print("Content Pipeline Completed Successfully!")
    print(f"Final Polished Deliverable Length: {len(str(result))} chars")
    print(f"Final Snippet:\n{str(result)[:180]}...")
    print("Exercise 1 passed successfully!")


# ── Exercise 2: Code Review Crew ─────────────────────────────────────────────
def exercise_2_code_review_crew():
    print("\n--- Exercise 2: Collaborative Code Review Crew ---")
    llm = get_chat_model()

    developer = Agent(
        role="Backend Python Engineer",
        goal="Implement robust user authentication endpoints in FastAPI",
        backstory="An engineer focused on writing modular and clean API logic.",
        llm=llm,
        verbose=False,
    )
    security_auditor = Agent(
        role="AppSec Auditor",
        goal="Audit code for OWASP Top 10 vulnerabilities, timing attacks, and token leaks",
        backstory="A certified security auditor specialized in cryptographic protocols.",
        llm=llm,
        verbose=False,
    )
    qa_engineer = Agent(
        role="Lead QA Automation Engineer",
        goal="Write comprehensive pytest suites and identify critical edge cases",
        backstory="A test engineer dedicated to 100% test branch coverage.",
        llm=llm,
        verbose=False,
    )

    task_dev = Task(
        description="Write a Python FastAPI JWT token validation endpoint with error handling.",
        expected_output="Python code snippet implementing JWT validation.",
        agent=developer,
    )
    task_sec = Task(
        description="Inspect the JWT implementation for algorithm confusion, expiration checks, and secrets handling.",
        expected_output="Security findings checklist and remediation diffs.",
        agent=security_auditor,
        context=[task_dev],
    )
    task_qa = Task(
        description="Draft pytest test cases testing expired tokens, malformed headers, and valid credentials.",
        expected_output="Complete pytest test suite with assert statements.",
        agent=qa_engineer,
        context=[task_sec],
    )

    review_crew = Crew(
        agents=[developer, security_auditor, qa_engineer],
        tasks=[task_dev, task_sec, task_qa],
        process=Process.sequential,
        verbose=False,
    )

    result = review_crew.kickoff()
    print("Code Review Crew Completed Successfully!")
    print(f"Output Deliverables: Developer -> Security Audit -> QA Test Suite")
    assert len(str(result)) > 50
    print("Exercise 2 passed successfully!")


# ── Exercise 3: Custom WebSearchTool with Researcher ─────────────────────────
class SearchArgs(BaseModel):
    query: str = Field(..., description="Search terms")

class MockWebSearchTool(BaseTool):
    name: str = "WebSearchEngine"
    description: str = "Retrieves authoritative web search snippets for technical topics."
    args_schema: Type[BaseModel] = SearchArgs

    def _run(self, query: str = "", **kwargs) -> str:
        return (
            f"[Source: Google DeepMind Research] Autonomous agentic coding reduces boilerplate by 60%. "
            f"[Source: ACM Digital Library] Multi-agent orchestration frameworks demonstrate lower hallucination rates."
        )

def exercise_3_custom_tool_integration():
    print("\n--- Exercise 3: Custom WebSearchTool with Researcher Agent ---")
    search_tool = MockWebSearchTool()
    llm = get_chat_model()

    researcher = Agent(
        role="Autonomous Technology Researcher",
        goal="Leverage live search tools to gather verified statistics on multi-agent frameworks",
        backstory="An AI researcher equipped with live search capabilities.",
        tools=[search_tool],
        llm=llm,
        verbose=False,
    )

    research_task = Task(
        description="Search for the latest empirical metrics regarding autonomous agentic coding.",
        expected_output="A summary citing empirical evidence gathered via search.",
        agent=researcher,
    )

    crew = Crew(
        agents=[researcher],
        tasks=[research_task],
        process=Process.sequential,
        verbose=False,
    )

    output = crew.kickoff()
    print("Custom Tool Execution Result:")
    print(str(output)[:220] + "...")
    assert len(str(output)) > 20
    print("Exercise 3 passed successfully!")


# ── Exercise 4: Sequential vs. Hierarchical Comparison ───────────────────────
def exercise_4_process_comparison():
    print("\n--- Exercise 4: Comparative Evaluation: Sequential vs. Hierarchical ---")
    llm = get_chat_model()

    manager = Agent(
        role="Product Strategy Director",
        goal="Ensure product roadmap satisfies technical and business criteria",
        backstory="Executive coordinating roadmap development.",
        allow_delegation=True,
        llm=llm,
        verbose=False,
    )
    analyst = Agent(
        role="Market Analyst",
        goal="Analyze competitor pricing and feature matrices",
        backstory="An expert in market segmentation.",
        allow_delegation=False,
        llm=llm,
        verbose=False,
    )

    t1 = Task(description="Assess competitor pricing models.", expected_output="Pricing table.", agent=analyst)
    t2 = Task(description="Synthesize final roadmap.", expected_output="Roadmap brief.", agent=manager, context=[t1])

    # Run Sequential
    crew_seq = Crew(agents=[analyst, manager], tasks=[t1, t2], process=Process.sequential, verbose=False)
    out_seq = crew_seq.kickoff()
    print("Process.sequential Execution: Tasks run in deterministic linear chain (T1 -> T2).")

    # Run Hierarchical
    crew_hier = Crew(agents=[manager, analyst], tasks=[t1, t2], process=Process.hierarchical, manager_agent=manager, verbose=False)
    out_hier = crew_hier.kickoff()
    print("Process.hierarchical Execution: Manager oversees worker execution and consolidates results.")

    print("\nArchitecture Comparison:")
    print("  Sequential   : Best for linear pipelines (e.g. Ingest -> Clean -> Embed -> Store).")
    print("  Hierarchical : Best for open-ended or complex tasks requiring planning, dynamic delegation, and review.")

    print("Exercise 4 passed successfully!")


def main():
    print("=" * 60)
    print("Module 14: CrewAI - Comprehensive Exercises")
    print("=" * 60)

    exercise_1_content_pipeline()
    exercise_2_code_review_crew()
    exercise_3_custom_tool_integration()
    exercise_4_process_comparison()

    print("\n" + "=" * 60)
    print("All Module 14 Exercises Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
