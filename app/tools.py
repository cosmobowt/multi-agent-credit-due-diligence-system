"""Tools the agents can call.

`web_search` is a real tool the research agent invokes autonomously. It uses
DuckDuckGo via the `ddgs` package, which needs no API key. The risk helper is a
deterministic function the risk analyst uses so the final score isn't left
entirely to the LLM's judgement — a pattern worth calling out in interviews
(LLM for reasoning, code for anything that must be reproducible).
"""
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for recent, factual information about a company:
    news, litigation, leadership, funding, defaults, or financial signals.
    Use specific queries like '<company> insolvency 2025' or '<company> revenue'.
    """
    try:
        from ddgs import DDGS

        # max_results is keyword-only; returns dicts with title/href/body.
        results = DDGS().text(query, max_results=4)
    except Exception as exc:  # keep the graph resilient
        return f"[Search unavailable for '{query}': {exc}]"

    if not results:
        return (
            f"No results for '{query}'. DuckDuckGo may be rate-limiting — "
            f"wait a few seconds and retry, or rephrase the query."
        )
    return "\n\n".join(
        f"- {r.get('title', 'Untitled')} ({r.get('href', '')})\n"
        f"  {r.get('body', '')[:500]}"
        for r in results
    )


# Weighted red-flag keywords. This is intentionally simple and transparent —
# the point is a reproducible signal the analyst can anchor on, then a place
# you can later swap in real financial ratios, bureau scores, etc.
RED_FLAGS = {
    "insolvency": 0.9, "bankruptcy": 0.9, "default": 0.8, "fraud": 0.95,
    "litigation": 0.5, "lawsuit": 0.5, "layoffs": 0.4, "downgrade": 0.6,
    "investigation": 0.7, "delisted": 0.8, "losses": 0.4, "debt": 0.3,
}
GREEN_FLAGS = {
    "profitable": -0.3, "funding": -0.2, "growth": -0.2, "expansion": -0.2,
    "award": -0.1, "partnership": -0.1, "record revenue": -0.3,
}


def compute_risk_score(text: str) -> float:
    """Map research text to a 0..1 risk score from weighted keyword signals.

    Replace this with a real model (bureau data, financial ratios, sentiment)
    when you extend the project — the interface (text -> float) stays the same.
    """
    if not isinstance(text, str):
        text = str(text)
    text_l = text.lower()
    score = 0.3  # neutral baseline
    for kw, w in {**RED_FLAGS, **GREEN_FLAGS}.items():
        if kw in text_l:
            score += w
    return max(0.0, min(1.0, score))