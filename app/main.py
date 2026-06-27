"""FastAPI layer over the agent graph.

Flow:
  POST /diligence/start    -> runs research..reviewer, pauses at the human
                              checkpoint, returns the draft memo + a thread_id.
  POST /diligence/{id}/approve -> resumes the paused run to finalize.
  GET  /diligence/{id}     -> inspects current state of a run.

The thread_id maps to a LangGraph checkpoint, so a paused run survives between
the two HTTP calls.
"""
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .graph import graph

load_dotenv()

app = FastAPI(title="Agentic Credit Due-Diligence")


class DiligenceRequest(BaseModel):
    company: str
    request: str = "a credit facility"
    max_revisions: int = 2


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@app.post("/diligence/start")
def start(req: DiligenceRequest):
    thread_id = str(uuid.uuid4())
    initial = {
        "company": req.company,
        "request": req.request,
        "revisions": 0,
        "max_revisions": req.max_revisions,
        "review_feedback": "",
    }
    # Runs until interrupt_before=["human_checkpoint"] pauses it.
    state = graph.invoke(initial, config=_config(thread_id))
    return {
        "thread_id": thread_id,
        "status": "awaiting_human_approval",
        "company": state["company"],
        "risk_score": state.get("risk_score"),
        "draft_memo": state.get("memo"),
        "revisions": state.get("revisions"),
    }


@app.post("/diligence/{thread_id}/approve")
def approve(thread_id: str):
    config = _config(thread_id)
    snapshot = graph.get_state(config)
    if not snapshot.next:
        raise HTTPException(404, "No paused run for this thread_id (or already finalized).")
    # Resuming with None continues from the saved checkpoint to END.
    final = graph.invoke(None, config=config)
    return {"thread_id": thread_id, "status": "finalized", "final_memo": final.get("memo")}


@app.get("/diligence/{thread_id}")
def get_state(thread_id: str):
    snapshot = graph.get_state(_config(thread_id))
    if not snapshot.values:
        raise HTTPException(404, "Unknown thread_id.")
    return {
        "thread_id": thread_id,
        "next": snapshot.next,
        "values": snapshot.values,
    }


@app.get("/")
def health():
    return {"ok": True, "service": "agentic-credit-diligence"}