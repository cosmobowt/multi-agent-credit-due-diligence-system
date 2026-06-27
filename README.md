# Multi-Agent Credit Due-Diligence System

An agentic AI system that performs autonomous credit due-diligence on a company.
Given a borrower name and a requested facility, a team of specialized LLM agents
researches the company on the live web, assesses its credit risk, drafts a
structured credit memo, and self-reviews that memo through a revision loop before
pausing for human approval.

Built with **LangGraph** for agent orchestration and **FastAPI** for the service layer.

---

## What it does

```
Input:  company = "Reliance Industries", request = "working capital line"
Output: a structured credit memo with a calibrated risk score and an
        Approve / Approve-with-conditions / Decline recommendation
```

Four specialized agents collaborate over a shared state, connected as a graph
with a self-correcting revision loop and a human approval gate:

```
            +----------------+
            |    research    |  ReAct agent: autonomous web-search tool use
            +-------+--------+
                    v
            +----------------+
            |      risk      |  LLM-based risk scoring (structured output)
            +-------+--------+
                    v
            +----------------+
            |     writer     |<-------------+
            +-------+--------+              |
                    v                       | "revise"
            +----------------+              |
            |    reviewer    |--------------+  (bounded by a revision cap)
            +-------+--------+
                    | "approve"
                    v
            +----------------+
            |human checkpoint|  run pauses here; resumes via API on approval
            +-------+--------+
                    v
               final memo
```

---

## Key engineering decisions

These are the design choices that make this an agentic system rather than a
single LLM call, and the patterns worth understanding in the code:

- **Cyclic graph with conditional routing.** After the reviewer runs, a routing
  function inspects the state and either sends the memo *back* to the writer to
  revise or forward to finalize. That backward edge makes the graph cyclic - the
  basis of self-correction. (`app/graph.py`, `route_after_review`)

- **Bounded self-correction loop.** A `max_revisions` counter caps the revise
  cycle so it always terminates, never looping forever.

- **Autonomous ReAct research agent.** The research agent reasons about what it
  needs, calls a web-search tool, reads the results, and decides whether to
  search again - overcoming the model's training cutoff with live data.
  (`app/agents.py`, `create_react_agent` + `web_search`)

- **Structured outputs for reliable routing.** The reviewer and risk agents
  return Pydantic models, not free text, so branching decisions are read from a
  validated schema instead of fragile string parsing.

- **Human-in-the-loop via checkpointing.** The graph compiles with
  `interrupt_before=["human_checkpoint"]` plus a checkpointer. The run persists
  its state when it pauses, so a separate API call can resume the exact same run
  later. (`app/graph.py`)

- **Provider-agnostic LLM layer with rate limiting.** Models are created through
  `init_chat_model`, so switching providers is one env var. A single *shared*
  token-bucket rate limiter paces all agents under free-tier request caps, since
  the quota is per-project. (`app/llm.py`)

- **LLM scorer over keyword scorer.** The risk score originally came from a
  keyword counter, which flagged a financially-strong borrower as maximum risk
  because real research text contains words like "debt" and "investigation" in
  neutral contexts. It was replaced with an LLM-based scorer that judges meaning,
  keeping the keyword score only as a transparency cross-check. (`app/agents.py`,
  `app/tools.py`)

---

## Architecture

| Layer | Responsibility | Files |
| --- | --- | --- |
| Service | HTTP API, request validation, pause/resume endpoints | `app/main.py` |
| Orchestration | Graph wiring, conditional routing, checkpointing | `app/graph.py` |
| Agents | The four agent node functions | `app/agents.py` |
| Tools | Web search, deterministic risk cross-check | `app/tools.py` |
| State | Shared typed state passed between agents | `app/state.py` |
| Model | Provider-agnostic LLM factory + rate limiter | `app/llm.py` |

---

## Tech stack

- **LangGraph** - agent orchestration as a stateful graph
- **FastAPI** - REST service layer with auto-generated docs
- **Pydantic** - request validation and structured LLM outputs
- **Google Gemini** - LLM provider (swappable via `init_chat_model`)
- **DuckDuckGo (`ddgs`)** - key-free web search for the research agent

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then add your key
```

Set your key in `.env`. The defaults run on Gemini's free tier:

```
LLM_MODEL=google_genai:gemini-2.5-flash-lite
LLM_RPM=6
GOOGLE_API_KEY=your-key-here
```

`LLM_RPM` paces requests to stay under the free-tier per-minute cap; lower it if
you hit rate limits, raise it on a paid tier. Get a free key from
[Google AI Studio](https://aistudio.google.com/apikey).

---

## Usage

### Quick test (no server)

Runs the full pipeline directly and prints each stage:

```bash
python test_run.py "Reliance Industries" "working capital line"
```

### As an API

```bash
uvicorn app.main:app --reload
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI, or use curl:

```bash
# Start a run - returns a draft memo, a risk score, and a thread_id, then pauses
curl -X POST localhost:8000/diligence/start \
  -H "Content-Type: application/json" \
  -d '{"company": "Reliance Industries", "request": "working capital line"}'

# Approve to finalize, using the returned thread_id
curl -X POST localhost:8000/diligence/<thread_id>/approve
```

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/diligence/start` | Runs the agents, pauses at the approval gate, returns the draft + `thread_id` |
| `POST` | `/diligence/{thread_id}/approve` | Resumes the paused run and finalizes it |
| `GET` | `/diligence/{thread_id}` | Inspects the saved state of a run |

The gap between `start` and `approve` is the human-in-the-loop pause - the run
stays frozen, its state persisted, until approval arrives.

---

## Notes and limitations

- **Research depth.** The research agent works from web-search results, not
  audited financials or filings. It demonstrates the agentic architecture; a
  production version would add tools for real financial data (statements, ratios,
  bureau scores) behind the same research interface.
- **In-memory checkpointing.** Paused runs are stored in memory (`MemorySaver`),
  so they don't survive a server restart. Swap for `SqliteSaver` / `PostgresSaver`
  to persist them.
- **Free-tier rate limits.** A full run makes several LLM calls; on a constrained
  free tier it is paced and therefore slow. `LLM_RPM` controls the pace.

---

## Possible extensions

- Real financial-data tools for the research agent
- An evaluation harness scoring memo quality across many companies
- Parallel research sub-agents (news / financials / legal) via graph fan-out
- Persistent storage so approvals commit to a database with an audit trail
- Structured API error handling (clean status codes instead of raw 500s)