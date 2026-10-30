"""Module 14 - CrewAI
Section 14.6: Hierarchical Process with Manager Agent

Demonstrates:
  - Configuring hierarchical multi-agent orchestration
  - Designating a Manager Agent with delegation authority
  - Coordinating worker agents dynamically under manager oversight
  - Evaluating final manager synthesis vs. individual worker contributions
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import get_chat_model

from crewai import Agent, Task, Crew, Process

def main():
    print("=" * 60)
    print("Module 14.6: Hierarchical Multi-Agent Crew with Manager")
    print("=" * 60)

    llm = get_chat_model()

    # 1. Manager Agent with delegation enabled
    manager = Agent(
        role="Engineering Project Director",
        goal="Orchestrate team execution and synthesize comprehensive system architectures",
        backstory="A seasoned director who delegates technical research and reviews all deliverables.",
        allow_delegation=True,
        llm=llm,
        verbose=True,
    )

    # 2. Worker Specialist Agents
    security_specialist = Agent(
        role="Security Architect",
        goal="Identify potential vulnerabilities, threat models, and mitigation strategies",
        backstory="An expert in zero-trust architectures and API security.",
        allow_delegation=False,
        llm=llm,
        verbose=True,
    )

    infra_specialist = Agent(
        role="Cloud Infrastructure Engineer",
        goal="Design high-availability cloud deployments with automated scaling",
        backstory="A cloud architect specializing in Kubernetes, Terraform, and multi-region failover.",
        allow_delegation=False,
        llm=llm,
        verbose=True,
    )

    # 3. Main Project Task overseen by Manager
    task_security = Task(
        description="Audit authentication, data-in-transit, and secrets management policies.",
        expected_output="Security audit report with risk ratings.",
        agent=security_specialist,
    )
    task_infra = Task(
        description="Design infrastructure topology with automated autoscaling and disaster recovery.",
        expected_output="Infrastructure specification with regional redundancy.",
        agent=infra_specialist,
    )
    main_task = Task(
        description="Consolidate the security audit and infrastructure specifications into a unified architectural proposal.",
        expected_output="Comprehensive System Architecture Proposal ready for C-suite review.",
        agent=manager,
        context=[task_security, task_infra],
    )

    # 4. Hierarchical Crew
    hierarchical_crew = Crew(
        agents=[manager, security_specialist, infra_specialist],
        tasks=[task_security, task_infra, main_task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True,
    )

    print("\nStarting Hierarchical Crew Kickoff...\n")
    final_output = hierarchical_crew.kickoff()

    print("\n" + "=" * 60)
    print("Hierarchical Crew Deliverable:")
    print("=" * 60)
    print(str(final_output))

    print("\nHierarchical manager workflow completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
