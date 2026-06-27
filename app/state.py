"""The shared state that flows through every node in the graph.

In LangGraph, each node receives the current state and returns a *partial*
update which is merged back in. Keeping all working memory in one typed dict
is what lets the agents collaborate without passing arguments around manually.
"""
from typing import Literal, TypedDict


class AgentState(TypedDict, total=False):
    # --- inputs ---
    company: str            # borrower / company name to assess
    request: str            # what the credit team wants (e.g. "₹50L working capital line")

    # --- produced by the research agent ---
    research: str           # raw findings: profile, news, signals

    # --- produced by the risk analyst agent ---
    risk_assessment: str    # narrative analysis of red/green flags
    risk_score: float       # 0.0 (safe) .. 1.0 (high risk)

    # --- produced by the memo writer agent ---
    memo: str               # the structured credit memo draft

    # --- produced by the reviewer agent ---
    review_feedback: str
    decision: Literal["approve", "revise"]

    # --- loop control ---
    revisions: int          # how many times the memo has been revised
    max_revisions: int      # hard cap so the revise loop always terminates