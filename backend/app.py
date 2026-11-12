"""FastAPI service for the maintenance ticket triage agent.
Run locally:  uvicorn backend.app:app --reload     then open http://localhost:8000
"""
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.triage import TriageAgent, load_data

app = FastAPI(title="AI Agent for Maintenance Ticket Prioritization")
BASE = Path(__file__).resolve().parents[1]
FRONTEND = BASE / "frontend" / "index.html"
EVAL_FILE = BASE / "evaluation_results.json"

agent = TriageAgent()  # trains the TF-IDF classifier once at startup (~1 s on the bundled data)


class Ticket(BaseModel):
    text: str = ""
    asset_type: str = "unknown"  # solar | wind | hydro | grid | unknown


@app.get("/")
def index():
    return FileResponse(FRONTEND)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/demo")
def demo():
    return {
        "case_study": "AI Agent for Maintenance Ticket Prioritization",
        "status": "running",
        "endpoints": ["POST /api/triage", "GET /api/samples", "GET /api/evaluation"],
    }


@app.post("/api/triage")
def triage(ticket: Ticket):
    if not ticket.text.strip():
        raise HTTPException(status_code=422, detail="Ticket text must not be empty.")
    return agent.run(ticket.text, ticket.asset_type)


@app.post("/api/run")  # kept for compatibility with the starter template
def run_case(ticket: Ticket):
    return triage(ticket)


@app.get("/api/samples")
def samples():
    df = load_data().head(8)
    return df[["ticket_id", "asset_type", "description"]].to_dict(orient="records")


@app.get("/api/evaluation")
def evaluation():
    if not EVAL_FILE.exists():
        raise HTTPException(status_code=404, detail="Run `python -m backend.evaluate` first.")
    return json.loads(EVAL_FILE.read_text())
