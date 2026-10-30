"""Module 13 - LangGraph & Agents
Section 13.4: Human-in-the-Loop & Persistent Memory

Demonstrates:
  - Using MemorySaver to persist agent state across sessions
  - Pausing execution before sensitive actions with interrupt_before
  - Inspecting checkpointed state using app.get_state()
  - Approving or editing state and resuming execution with app.invoke(None, config)
"""
import sys
from pathlib import Path
from typing import TypedDict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 1. State definition
class TransferState(TypedDict):
    sender: str
    recipient: str
    amount: float
    status: str
    approval_required: bool
    audit_trail: List[str]


# 2. Nodes
def validate_node(state: TransferState) -> dict:
    amt = state["amount"]
    needs_approval = amt > 1000.0
    return {
        "status": "pending_approval" if needs_approval else "auto_approved",
        "approval_required": needs_approval,
        "audit_trail": state.get("audit_trail", []) + [f"Validated transfer of ${amt:.2f}. Needs approval: {needs_approval}"],
    }


def execute_transfer_node(state: TransferState) -> dict:
    return {
        "status": "completed",
        "audit_trail": state.get("audit_trail", []) + [f"Transferred ${state['amount']:.2f} from {state['sender']} to {state['recipient']}."],
    }


def main():
    print("=" * 60)
    print("Module 13.4: Human-in-the-Loop & Checkpoint Memory")
    print("=" * 60)

    # 3. Build Graph with Checkpointer
    checkpointer = MemorySaver()
    workflow = StateGraph(TransferState)

    workflow.add_node("validate", validate_node)
    workflow.add_node("execute_transfer", execute_transfer_node)

    workflow.add_edge(START, "validate")
    workflow.add_edge("validate", "execute_transfer")
    workflow.add_edge("execute_transfer", END)

    # Interrupt before the sensitive execution node!
    app = workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute_transfer"],
    )

    thread_config = {"configurable": {"thread_id": "transfer-session-101"}}

    # Step 1: Initiate a large financial transfer
    initial_request = {
        "sender": "Alice",
        "recipient": "Vendor Corp",
        "amount": 5000.0,
        "status": "initiated",
        "approval_required": False,
        "audit_trail": ["Transaction initiated by client."],
    }

    print(f"Submitting Transfer: ${initial_request['amount']} from {initial_request['sender']} to {initial_request['recipient']}")
    _ = app.invoke(initial_request, config=thread_config)

    # Step 2: Inspect paused state at breakpoint
    current_state = app.get_state(thread_config)
    print("\n[PAUSED AT BREAKPOINT]")
    print(f"Next pending node  : {current_state.next}")
    print(f"Transaction Status : {current_state.values.get('status')}")
    print(f"Approval Required  : {current_state.values.get('approval_required')}")
    print("Audit Log so far   :")
    for log in current_state.values.get("audit_trail", []):
        print(f"  - {log}")

    # Step 3: Simulate Human Approval
    print("\n[HUMAN REVIEWER]: Action Approved. Resuming execution...")
    resumed_state = app.invoke(None, config=thread_config)

    # Step 4: Final State Inspection
    print("\n[TRANSACTION COMPLETED]")
    print(f"Final Status: {resumed_state['status']}")
    print("Complete Audit Trail:")
    for log in resumed_state["audit_trail"]:
        print(f"  - {log}")

    print("\nHuman-in-the-Loop workflow completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
