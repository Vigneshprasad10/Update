"""
FastAPI server for the Support AI Agent.

Endpoints:
  POST /api/tickets/analyze         — submit ticket, returns job_id
  GET  /api/tickets/{job_id}        — poll full job state
  GET  /api/tickets                 — list all jobs
  WS   /ws/tickets/{job_id}         — real-time WebSocket stream
  GET  /api/stream/{job_id}         — SSE stream (EventSource)
  GET  /health                      — health check
  GET  /                            — serves dashboard.html
"""

import asyncio
import json
import logging
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel

from agents.support_agent import run_support_agent
from pdf_generator import generate_pdf_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

app = FastAPI(
    title="Support AI Agent API",
    description="AI-powered support ticket enrichment via LangGraph + Gemini",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Stores ────────────────────────────────────────────────────────────────────
job_store: dict[str, dict] = {}
job_subscribers: dict[str, list] = {}   # job_id -> [asyncio.Queue, ...]


# ── Models ────────────────────────────────────────────────────────────────────

class TicketRequest(BaseModel):
    ticket_summary: str
    ticket_description: str
    priority: str = "high"
    product_version: str = ""
    environment: str = "production"
    customer_id: str = ""
    tags: list[str] = []


class TicketResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ── Broadcast helpers ─────────────────────────────────────────────────────────

def _push_event(job_id: str, event: dict):
    """Thread-safe broadcast to all subscriber queues."""
    for q in job_subscribers.get(job_id, []):
        try:
            loop = getattr(q, "_loop", None)
            if loop and loop.is_running():
                loop.call_soon_threadsafe(q.put_nowait, event)
            else:
                q.put_nowait(event)
        except Exception:
            pass


def _update_job(job_id: str, patch: dict):
    job_store[job_id].update(patch)
    _push_event(job_id, {**job_store[job_id], "event_type": "state_update"})


def _emit(job_id: str, step: str, message: str, extra: dict = None):
    event = {
        "event_type": "progress",
        "job_id": job_id,
        "step": step,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
        **(extra or {}),
    }
    job_store[job_id].setdefault("events", []).append(event)
    _push_event(job_id, event)


# ── Background worker ─────────────────────────────────────────────────────────

def process_ticket_sync(job_id: str, request: TicketRequest):
    """Run agent in background thread, emitting granular progress events."""
    try:
        _update_job(job_id, {"status": "processing",
                              "started_at": datetime.utcnow().isoformat()})
        _emit(job_id, "start", "Agent initializing…")

        import agents.support_agent as am

        orig_extract    = am.extract_intent_node
        orig_jira       = am.query_jira_node
        orig_confluence = am.query_confluence_node
        orig_github     = am.query_github_node
        orig_synth      = am.synthesize_and_resolve_node
        orig_ticket     = am.create_ticket_node

        def w_extract(state):
            _emit(job_id, "extract_intent", "Analyzing ticket with Gemini…")
            r = orig_extract(state)
            _emit(job_id, "extract_intent_done", "Keywords extracted", {
                "keywords": r.get("keywords", []),
                "error_codes": r.get("error_codes", []),
                "affected_components": r.get("affected_components", []),
            })
            return r

        def w_jira(state):
            _emit(job_id, "query_jira", "Searching Jira for known bugs…")
            r = orig_jira(state)
            ctx = r.get("jira_context", {})
            _emit(job_id, "query_jira_done",
                  f"Found {len(ctx.get('known_bugs',[]))} bugs, "
                  f"{len(ctx.get('related_issues',[]))} related tickets",
                  {"jira_context": ctx})
            return r

        def w_confluence(state):
            _emit(job_id, "query_confluence", "Searching Confluence KB & runbooks…")
            r = orig_confluence(state)
            ctx = r.get("confluence_context", {})
            _emit(job_id, "query_confluence_done",
                  f"Found {len(ctx.get('relevant_pages',[]))} articles, "
                  f"{len(ctx.get('runbooks',[]))} runbooks",
                  {"confluence_context": ctx})
            return r

        def w_github(state):
            _emit(job_id, "query_github", "Scanning recent commits & PRs…")
            r = orig_github(state)
            ctx = r.get("github_context", {})
            _emit(job_id, "query_github_done",
                  f"Found {len(ctx.get('recent_commits',[]))} commits, "
                  f"{len(ctx.get('open_prs',[]))} PRs",
                  {"github_context": ctx})
            return r

        def w_synth(state):
            _emit(job_id, "synthesize", "Synthesizing context with Gemini…")
            r = orig_synth(state)
            res = r.get("resolution", {})
            _emit(job_id, "synthesize_done",
                  f"Analysis complete — confidence "
                  f"{int(res.get('confidence_score', 0) * 100)}%",
                  {"resolution": res, "final_summary": r.get("final_summary", "")})
            return r

        def w_ticket(state):
            # Emit synthesize result first so UI shows full analysis
            resolution = state.get("resolution", {})
            enriched = state.get("enriched_ticket", {})
 
            # Now pause for human approval
            _emit(job_id, "awaiting_approval", "⏳ Awaiting human approval before creating ticket…",
                  {"enriched_ticket": enriched,
                   "resolution": resolution})
            _update_job(job_id, {"status": "awaiting_approval"})
 
            # Wait for approval (max 5 minutes)
            import time
            timeout = 300
            elapsed = 0
            while elapsed < timeout:
                decision = job_store[job_id].get("human_decision")
                if decision == "approve":
                    _emit(job_id, "create_ticket", "✅ Approved! Creating enriched Jira ticket…")
                    r = orig_ticket(state)
                    _emit(job_id, "create_ticket_done", "Enriched ticket created",
                          {"jira_context": r.get("jira_context", {})})
                    return r
                elif decision == "reject":
                    _emit(job_id, "ticket_rejected", "❌ Ticket creation rejected by human reviewer.")
                    return {"steps_completed": ["create_ticket"], "messages": []}
                time.sleep(1)
                elapsed += 1
 
            _emit(job_id, "ticket_timeout", "⏰ Approval timeout — ticket creation skipped.")
            return {"steps_completed": ["create_ticket"], "messages": []}
 

        am.extract_intent_node         = w_extract
        am.query_jira_node             = w_jira
        am.query_confluence_node       = w_confluence
        am.query_github_node           = w_github
        am.synthesize_and_resolve_node = w_synth
        am.create_ticket_node          = w_ticket
        am.support_agent = am.build_support_agent_graph()

        try:
            final_state = run_support_agent(
                ticket_summary=request.ticket_summary,
                ticket_description=request.ticket_description,
                priority=request.priority,
                product_version=request.product_version,
                environment=request.environment,
                customer_id=request.customer_id,
                tags=request.tags,
            )
        finally:
            am.extract_intent_node         = orig_extract
            am.query_jira_node             = orig_jira
            am.query_confluence_node       = orig_confluence
            am.query_github_node           = orig_github
            am.synthesize_and_resolve_node = orig_synth
            am.create_ticket_node          = orig_ticket
            am.support_agent = am.build_support_agent_graph()

        result_payload = {
            "steps_completed":     final_state.get("steps_completed", []),
            "keywords":            final_state.get("keywords", []),
            "error_codes":         final_state.get("error_codes", []),
            "affected_components": final_state.get("affected_components", []),
            "jira_context":        final_state.get("jira_context", {}),
            "confluence_context":  final_state.get("confluence_context", {}),
            "github_context":      final_state.get("github_context", {}),
            "resolution":          final_state.get("resolution", {}),
            "final_summary":       final_state.get("final_summary", ""),
            "enriched_ticket":     final_state.get("enriched_ticket", {}),
            "ticket_id":           (final_state.get("jira_context") or {}).get("ticket_id"),
            "errors":              final_state.get("errors", []),
        }

        _update_job(job_id, {
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "result": result_payload,
        })
        _push_event(job_id, {"event_type": "done", "job_id": job_id,
                              "result": result_payload})

    except Exception as exc:
        logger.error(f"[JOB {job_id}] Failed: {exc}", exc_info=True)
        _update_job(job_id, {
            "status": "failed",
            "error": str(exc),
            "completed_at": datetime.utcnow().isoformat(),
        })
        _push_event(job_id, {"event_type": "error", "job_id": job_id,
                              "error": str(exc)})


# ── REST endpoints ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = BASE_DIR / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse("<h1>dashboard.html not found next to server.py</h1>", 404)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "support-ai-agent",
        "timestamp": datetime.utcnow().isoformat(),
        "jobs_total": len(job_store),
        "jobs_processing": sum(1 for j in job_store.values()
                                if j.get("status") == "processing"),
    }


@app.get("/api/reports/{filename}")
async def download_report(filename: str):
    from fastapi.responses import FileResponse
    filepath = os.path.join("reports", filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "Report not found"})
    return FileResponse(filepath, media_type="application/pdf", filename=filename)


@app.get("/api/reports")
async def list_reports(job_id: str = None):
    """Return PDF for specific job, or all reports if no job_id provided"""
    if job_id and job_id in job_store:
        # Return only the PDF for this specific job
        pdf_path = job_store[job_id].get("pdf_path")
        if pdf_path and os.path.exists(pdf_path):
            return {"reports": [os.path.basename(pdf_path)]}
    
    # Fallback: return all reports (for backward compatibility)
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return {"reports": []}
    reports = [f.name for f in reports_dir.glob("*.pdf")]
    reports.sort(reverse=True)
    return {"reports": reports}


@app.post("/api/tickets/analyze", response_model=TicketResponse)
async def analyze_ticket(request: TicketRequest):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "events": [],
        "request": request.model_dump(),
        "created_at": datetime.utcnow().isoformat(),
    }
    job_subscribers[job_id] = []
    threading.Thread(target=process_ticket_sync, args=(job_id, request),
                     daemon=True).start()
    return TicketResponse(
        job_id=job_id,
        status="queued",
        message=f"Analysis started — WS: /ws/tickets/{job_id}",
    )


@app.get("/api/tickets/{job_id}")
async def get_ticket_status(job_id: str):
    if job_id not in job_store:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job_store[job_id]

@app.post("/api/tickets/{job_id}/approve")
async def approve_ticket(job_id: str):
    if job_id not in job_store:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    job_store[job_id]["human_decision"] = "approve"
    
    # Regenerate PDF on approval with the latest ticket ID
    try:
        result = job_store[job_id].get("result", {})
        if result:
            import copy
            result_copy = copy.deepcopy(result)
            
            # Ensure we have the latest ticket_id from the result
            if "jira_context" not in result_copy:
                result_copy["jira_context"] = {}
            
            pdf_path = generate_pdf_report(result_copy)
            # Store PDF path in job_store so we can return the correct one
            job_store[job_id]["pdf_path"] = pdf_path
            logger.info(f"[PDF] Report regenerated on approval with ticket ID: {result_copy.get('jira_context', {}).get('ticket_id')}")
    except Exception as e:
        logger.warning(f"[PDF] Report generation failed on approval: {e}")

    _push_event(job_id, {"event_type": "approved", "job_id": job_id})

    return {"status": "approved", "job_id": job_id}
 
 
@app.post("/api/tickets/{job_id}/reject")
async def reject_ticket(job_id: str):
    if job_id not in job_store:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    job_store[job_id]["human_decision"] = "reject"
    
    # Generate PDF on rejection - always show "Not created" regardless of prior approval
    try:
        result = job_store[job_id].get("result", {})
        if result:
            import copy
            result_copy = copy.deepcopy(result)
            
            # ALWAYS set Jira Ticket to "Not created" on rejection
            # This covers both cases: rejection without approval, and rejection after approval
            if "jira_context" not in result_copy:
                result_copy["jira_context"] = {}
            
            # Force ticket_id to "Not created" - this overrides any previously created ticket
            result_copy["jira_context"]["ticket_id"] = "Not created"
            
            pdf_path = generate_pdf_report(result_copy)
            # Store PDF path in job_store so we can return the correct one
            job_store[job_id]["pdf_path"] = pdf_path
            logger.info(f"[PDF] Report generated on rejection: {pdf_path}")
    except Exception as e:
        logger.warning(f"[PDF] Report generation failed on rejection: {e}")

    _push_event(job_id, {"event_type": "rejected", "job_id": job_id})

    return {"status": "rejected", "job_id": job_id}
 
@app.get("/api/tickets")
async def list_tickets():
    return {
        "count": len(job_store),
        "tickets": [
            {
                "job_id": k,
                "status": v.get("status"),
                "summary": v.get("request", {}).get("ticket_summary", "")[:80],
                "priority": v.get("request", {}).get("priority", ""),
                "created_at": v.get("created_at"),
                "completed_at": v.get("completed_at"),
                "ticket_id": (v.get("result") or {}).get("ticket_id"),
            }
            for k, v in sorted(job_store.items(),
                                key=lambda x: x[1].get("created_at", ""),
                                reverse=True)
        ],
    }


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws/tickets/{job_id}")
async def ticket_websocket(websocket: WebSocket, job_id: str):
    await websocket.accept()
    if job_id not in job_store:
        await websocket.send_json({"event_type": "error", "error": "Job not found"})
        await websocket.close()
        return

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    queue._loop = loop
    job_subscribers.setdefault(job_id, []).append(queue)

    for evt in job_store[job_id].get("events", []):
        await websocket.send_json(evt)

    if job_store[job_id].get("status") in ("completed", "failed"):
        await websocket.send_json({**job_store[job_id], "event_type": "done"})
        job_subscribers[job_id].remove(queue)
        return

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30)
                await websocket.send_json(event)
                if event.get("event_type") in ("done", "error"):
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({"event_type": "heartbeat"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"event_type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        subs = job_subscribers.get(job_id, [])
        if queue in subs:
            subs.remove(queue)


# ── SSE endpoint ──────────────────────────────────────────────────────────────

@app.get("/api/stream/{job_id}")
async def sse_stream(job_id: str, request: Request):
    if job_id not in job_store:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    queue._loop = loop
    job_subscribers.setdefault(job_id, []).append(queue)

    async def generate() -> AsyncGenerator[str, None]:
        for evt in job_store[job_id].get("events", []):
            yield f"data: {json.dumps(evt)}\n\n"
        if job_store[job_id].get("status") in ("completed", "failed"):
            yield f"data: {json.dumps({**job_store[job_id], 'event_type': 'done'})}\n\n"
            return
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("event_type") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    yield 'data: {"event_type":"heartbeat"}\n\n'
        finally:
            subs = job_subscribers.get(job_id, [])
            if queue in subs:
                subs.remove(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
