"""Wire the agents into a LangGraph state machine.

Two patterns worth understanding here:

1. Conditional edge after the reviewer — this is what makes the graph *cyclic*.
   The reviewer can route control back to the writer ("revise") or forward
   ("approve"). A revision counter caps the loop so it always terminates.

2. interrupt_before=["human_checkpoint"] + a checkpointer — execution pauses
   before the human node and persists state, so a separate API call (a human
   approving) can resume the exact same run later. That's true
   human-in-the-loop, not a fake `input()` prompt.
"""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .agents import research_node, reviewer_node, risk_node, writer_node
from .state import AgentState


def route_after_review(state: AgentState) -> str:
    """Send the memo back to the writer to revise, or forward once approved /
    once the revision cap is hit."""
    if state.get("decision") == "approve":
        return "approved"
    if state.get("revisions", 0) >= state.get("max_revisions", 2):
        return "approved"  # give up revising; let the human decide
    return "revise"


def human_checkpoint(state: AgentState) -> dict:
    """No-op node. Its value is the interrupt *before* it — the run pauses here
    so a human can inspect the memo and approve via the API before finalizing."""
    return {}


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("research", research_node)
    g.add_node("risk", risk_node)
    g.add_node("writer", writer_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("human_checkpoint", human_checkpoint)

    g.add_edge(START, "research")
    g.add_edge("research", "risk")
    g.add_edge("risk", "writer")
    g.add_edge("writer", "reviewer")
    g.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"revise": "writer", "approved": "human_checkpoint"},
    )
    g.add_edge("human_checkpoint", END)

    # MemorySaver keeps state in-process. Swap for SqliteSaver / PostgresSaver
    # to persist runs across restarts.
    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer, interrupt_before=["human_checkpoint"])


graph = build_graph()