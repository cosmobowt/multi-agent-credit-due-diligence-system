"""Smoke test: run the full agent graph directly, no API involved.

This is the fastest way to confirm your key works and the agents collaborate.
Run it before testing the HTTP layer so you know which layer a bug is in.

    python test_run.py "Reliance Industries" "working capital line"
"""
import sys

from dotenv import load_dotenv

from app.graph import graph

load_dotenv()


def main():
    company = sys.argv[1] if len(sys.argv) > 1 else "Tata Consultancy Services"
    request = sys.argv[2] if len(sys.argv) > 2 else "a credit facility"

    config = {"configurable": {"thread_id": "smoke-test"}}
    initial = {
        "company": company,
        "request": request,
        "revisions": 0,
        "max_revisions": 1,
        "review_feedback": "",
    }

    print(f"\n=== Running diligence on: {company} ===\n")

    # Stream each node as it finishes so you can watch the agents work.
    for step in graph.stream(initial, config=config):
        for node_name, update in step.items():
            print(f"--- {node_name} done ---")
            if "risk_score" in update:
                print(f"    risk_score: {update['risk_score']:.2f}")
            if "decision" in update:
                print(f"    reviewer decision: {update['decision']}")

    # The run pauses before the human checkpoint; inspect the draft.
    state = graph.get_state(config).values
    print(f"\n=== DRAFT MEMO (revisions: {state.get('revisions')}) ===\n")
    print(state.get("memo", "[no memo produced]"))

    # Simulate the human approving to finalize.
    print("\n=== Approving (resuming to END) ===")
    final = graph.invoke(None, config=config)
    print("Finalized. Status: done.")


if __name__ == "__main__":
    main()