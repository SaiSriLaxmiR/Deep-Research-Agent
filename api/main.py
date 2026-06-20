"""
api/main.py

FastAPI wrapper around the deep research graph.

Endpoints:
    POST /research          — start a new research run
    GET  /research/{id}     — get status and result of a run
    POST /research/{id}/approve — approve and resume a paused run
    GET  /research          — list all recent research runs

Run with:
    uvicorn api.main:app --reload --port 8000

Then test with:
    curl -X POST http://localhost:8000/research \
         -H "Content-Type: application/json" \
         -d '{"query": "How is AI changing healthcare?", "depth": "quick"}'
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.graph import build_graph, run_research
from core.state import initial_state
from core.checkpointer import get_checkpointer, get_thread_config, generate_thread_id


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Deep Research Agent",
    description="Multi-agent LangGraph research system powered by Groq",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-memory job tracker ─────────────────────────────────────────────────────
# Tracks all research runs: their thread_id, status, and result.
# In production: store this in the same PostgreSQL database.

jobs: dict = {}


# ── Request / Response models ─────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str
    depth: str = "standard"         # "quick" | "standard" | "deep"
    human_review: bool = False      # pause before writing for approval


class ResearchResponse(BaseModel):
    job_id: str
    thread_id: str
    status: str
    message: str


class ResearchResult(BaseModel):
    job_id: str
    thread_id: str
    query: str
    status: str
    quality_score: float
    total_sources: int
    iterations: int
    final_report: Optional[str]
    report_sections: list
    created_at: str
    needs_approval: bool = False


class ApproveRequest(BaseModel):
    approved: bool


# ── Background task — runs the graph without blocking the API ─────────────────

def run_research_job(job_id: str, query: str, depth: str, human_review: bool):
    """
    Runs in the background so the API returns immediately.
    Updates jobs[job_id] as the graph progresses.
    """
    try:
        jobs[job_id]["status"] = "running"

        graph = build_graph(human_review=human_review)
        thread_id = jobs[job_id]["thread_id"]
        config = get_thread_config(thread_id)
        state = initial_state(query, depth=depth)

        if human_review:
            # Run until pause point
            graph.invoke(state, config=config)

            # Check if paused before writer
            current = graph.get_state(config)
            if "writer" in (current.next or []):
                jobs[job_id]["status"] = "awaiting_approval"
                jobs[job_id]["merged_context"] = current.values.get("merged_context", "")
                jobs[job_id]["quality_score"] = current.values.get("quality_score", 0.0)
                return
        else:
            result = graph.invoke(state, config=config)
            jobs[job_id]["status"] = result.get("status", "complete")
            jobs[job_id]["result"] = result

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        print(f"[api] job {job_id} failed: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Start a new research run.
    Returns immediately with a job_id — use GET /research/{job_id} to poll for results.
    """
    job_id = str(uuid.uuid4())[:8]
    thread_id = generate_thread_id(request.query)

    jobs[job_id] = {
        "job_id": job_id,
        "thread_id": thread_id,
        "query": request.query,
        "depth": request.depth,
        "human_review": request.human_review,
        "status": "queued",
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
    }

    background_tasks.add_task(
        run_research_job,
        job_id=job_id,
        query=request.query,
        depth=request.depth,
        human_review=request.human_review,
    )

    print(f"[api] started job {job_id} for query: '{request.query[:50]}...'")

    return ResearchResponse(
        job_id=job_id,
        thread_id=thread_id,
        status="queued",
        message=f"Research started. Poll GET /research/{job_id} for results.",
    )


@app.get("/research/{job_id}", response_model=ResearchResult)
async def get_research(job_id: str):
    """
    Get the status and result of a research run.
    Poll this endpoint every few seconds until status is 'complete'.

    Status values:
        queued            — waiting to start
        running           — agents are working
        awaiting_approval — paused before writer, needs human approval
        complete          — final report ready
        error             — something went wrong
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]
    result = job.get("result") or {}

    return ResearchResult(
        job_id=job_id,
        thread_id=job["thread_id"],
        query=job["query"],
        status=job["status"],
        quality_score=result.get("quality_score", 0.0),
        total_sources=result.get("total_sources", 0),
        iterations=result.get("iteration", 0),
        final_report=result.get("final_report"),
        report_sections=result.get("report_sections", []),
        created_at=job["created_at"],
        needs_approval=job["status"] == "awaiting_approval",
    )


@app.post("/research/{job_id}/approve")
async def approve_research(
    job_id: str,
    request: ApproveRequest,
    background_tasks: BackgroundTasks
):
    """
    Approve or reject a paused research run.
    Only valid when status is 'awaiting_approval'.

    If approved=True:  graph resumes and generates the final report
    If approved=False: job is cancelled
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs[job_id]

    if job["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not awaiting approval (status: {job['status']})"
        )

    if not request.approved:
        jobs[job_id]["status"] = "cancelled"
        return {"message": "Research cancelled."}

    # Resume the graph in the background
    def resume_job(job_id: str):
        try:
            graph = build_graph(human_review=True)
            thread_id = jobs[job_id]["thread_id"]
            config = get_thread_config(thread_id)
            result = graph.invoke(None, config=config)
            jobs[job_id]["status"] = result.get("status", "complete")
            jobs[job_id]["result"] = result
        except Exception as e:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)

    jobs[job_id]["status"] = "running"
    background_tasks.add_task(resume_job, job_id=job_id)

    return {"message": "Research approved. Generating final report..."}


@app.get("/research")
async def list_research():
    """List all research jobs with their current status."""
    return [
        {
            "job_id": j["job_id"],
            "query": j["query"][:60] + "..." if len(j["query"]) > 60 else j["query"],
            "status": j["status"],
            "created_at": j["created_at"],
        }
        for j in jobs.values()
    ]


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "active_jobs": len([j for j in jobs.values() if j["status"] == "running"]),
        "total_jobs": len(jobs),
    }


# ── Dev entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)