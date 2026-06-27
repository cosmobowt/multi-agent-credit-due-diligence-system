"""The agents, one per graph node.

Each function takes the full state and returns a partial update. This is the
heart of the project — most of your own engineering effort (better prompts,
structured outputs, more tools, evaluation) goes here.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState
from .tools import compute_risk_score, web_search


# ---------------------------------------------------------------------------
# 1. Research agent — an autonomous sub-agent with a web-search tool.
#    create_react_agent gives it a reason/act loop: it decides what to search,
#    reads results, and searches again until it has enough to summarise.
# ---------------------------------------------------------------------------
_research_agent = create_react_agent(
    get_llm(temperature=0.0),
    tools=[web_search],
    prompt=(
        "You are a credit research analyst. Investigate the company using web "
        "search: business profile, recent news, leadership, legal issues, and "
        "signals of financial distress or strength. Run AT MOST 2 focused "
        "searches, then stop and write a concise factual briefing. Do not keep "
        "searching once you have a reasonable picture."
    ),
)


def _to_text(content) -> str:
    """Normalize an LLM message's content to a plain string.

    Most providers return a string, but Gemini can return a list of content
    parts (dicts with a 'text' key, or plain strings). This flattens both.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def research_node(state: AgentState) -> dict:
    task = (
        f"Investigate '{state['company']}'. The credit team is considering: "
        f"{state.get('request', 'a credit facility')}. Gather everything relevant."
    )
    result = _research_agent.invoke({"messages": [HumanMessage(content=task)]})
    findings = _to_text(result["messages"][-1].content)
    return {"research": findings}


# ---------------------------------------------------------------------------
# 2. Risk analyst agent — reads the research and produces a STRUCTURED risk
#    score with justification. The deterministic keyword scorer is kept only as
#    a transparency cross-check, not the source of truth: real research text is
#    full of words like "debt" and "investigation" in neutral or positive
#    contexts, which a naive counter mis-reads as high risk. The LLM judges
#    meaning; we surface any divergence for the reader.
# ---------------------------------------------------------------------------
class RiskAssessment(BaseModel):
    risk_score: float = Field(
        description="0.0 = very safe borrower, 1.0 = very high credit risk"
    )
    red_flags: list[str] = Field(description="Specific concerns found in the research")
    green_flags: list[str] = Field(description="Specific mitigating strengths")
    rationale: str = Field(description="Two or three sentences of overall judgement")


def risk_node(state: AgentState) -> dict:
    research = state.get("research", "")
    keyword_score = compute_risk_score(research)  # cross-check only

    llm = get_llm(temperature=0.1).with_structured_output(RiskAssessment)
    result: RiskAssessment = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a senior credit risk analyst. Read the research and assess "
                    "the borrower's credit risk. Judge the *meaning* of findings: the mere "
                    "presence of words like 'debt' or 'investigation' is not itself a red "
                    "flag — context decides. Return a calibrated risk_score in [0,1], the "
                    "concrete red and green flags, and a short rationale."
                )
            ),
            HumanMessage(
                content=f"Company: {state['company']}\n\nResearch findings:\n{research}"
            ),
        ]
    )

    # Clamp defensively in case the model returns out-of-range.
    score = max(0.0, min(1.0, float(result.risk_score)))

    # Build the narrative the writer will consume.
    parts = [result.rationale.strip(), ""]
    if result.red_flags:
        parts.append("Red flags:")
        parts += [f"- {f}" for f in result.red_flags]
        parts.append("")
    if result.green_flags:
        parts.append("Green flags:")
        parts += [f"- {f}" for f in result.green_flags]
        parts.append("")
    # Surface divergence between the LLM score and the naive keyword score.
    if abs(score - keyword_score) >= 0.3:
        parts.append(
            f"(Note: naive keyword score was {keyword_score:.2f} vs analyst "
            f"score {score:.2f}; keyword signal flagged unreliable here.)"
        )
    assessment = "\n".join(parts).strip()

    return {"risk_assessment": assessment, "risk_score": score}


# ---------------------------------------------------------------------------
# 3. Memo writer agent — drafts the structured credit memo. On a revise loop
#    it incorporates the reviewer's feedback instead of starting fresh.
# ---------------------------------------------------------------------------
def writer_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.3)
    revision_note = ""
    if state.get("review_feedback"):
        revision_note = (
            "\n\nThis is a REVISION. Address this reviewer feedback directly:\n"
            f"{state['review_feedback']}\n\nPrevious draft:\n{state.get('memo', '')}"
        )
    msgs = [
        SystemMessage(
            content=(
                "You are a credit officer writing a formal credit memo. Use these "
                "sections: 1) Borrower Overview 2) Requested Facility 3) Key Findings "
                "4) Risk Assessment 5) Recommendation (Approve / Approve with "
                "conditions / Decline) with a one-line rationale. Be specific and concise."
            )
        ),
        HumanMessage(
            content=(
                f"Company: {state['company']}\n"
                f"Request: {state.get('request', 'N/A')}\n"
                f"Analyst risk score (0=safe, 1=high risk): "
                f"{state.get('risk_score', 0):.2f}\n\n"
                f"Research:\n{state.get('research', '')}\n\n"
                f"Risk analysis:\n{state.get('risk_assessment', '')}"
                f"{revision_note}"
            )
        ),
    ]
    memo = _to_text(llm.invoke(msgs).content)
    return {"memo": memo, "revisions": state.get("revisions", 0) + 1}


# ---------------------------------------------------------------------------
# 4. Reviewer agent — critiques the memo and returns a STRUCTURED decision.
#    with_structured_output forces clean approve/revise routing — no fragile
#    string parsing.
# ---------------------------------------------------------------------------
class Review(BaseModel):
    decision: str = Field(description="'approve' or 'revise'")
    feedback: str = Field(description="Specific, actionable critique. Empty if approving.")


def reviewer_node(state: AgentState) -> dict:
    llm = get_llm(temperature=0.0).with_structured_output(Review)
    msgs = [
        SystemMessage(
            content=(
                "You are a meticulous credit committee reviewer. Approve the memo only "
                "if it is well-supported, internally consistent, and the recommendation "
                "matches the risk. Otherwise return 'revise' with concrete fixes."
            )
        ),
        HumanMessage(content=f"Review this credit memo:\n\n{state['memo']}"),
    ]
    review: Review = llm.invoke(msgs)
    decision = "approve" if review.decision.lower().startswith("approve") else "revise"
    return {"decision": decision, "review_feedback": review.feedback}