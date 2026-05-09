"""
NHTSA Project — FastAPI backend.

What this file does
-------------------
This is the ASGI entrypoint that uvicorn imports as `main:app` on Render.
It exposes the LangGraph NHTSA complaint-analysis pipeline as an HTTP API:

    GET  /             -> service banner (sanity check)
    GET  /health       -> 200 OK probe for Render's healthcheck
    POST /run          -> Server-Sent Events stream of pipeline progress

The pipeline itself lives in `pipeline_runner.py`, which sets the headless
runtime flag and imports the compiled LangGraph app. We keep the FastAPI
wiring (routes, CORS, request models) in this file and the LangGraph wiring
in pipeline_runner.py so the two concerns are separable when reading or
debugging.

Why CORS lives here
-------------------
The frontend is hosted on Vercel. Browsers enforce the Same-Origin Policy,
which means JavaScript running on `https://<vercel-app>.vercel.app` cannot
read the response body of an XHR / fetch / EventSource request to a
different origin (this Render URL) unless the Render server explicitly
opts in via CORS response headers.

CORS is a SERVER concern, not a client concern: Vercel does not need to
configure anything; this FastAPI app does. The `allow_origins` list below
is the allowlist of caller origins this server will accept, NOT a list of
where this server can be hosted. Adding the Render URL itself to that list
would do nothing useful, since browsers stamp the request's `Origin` with
the *frontend's* domain.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# pipeline_runner sets the headless flag and imports the compiled graph at
# module load. Importing it here means any startup-time failure (bad path,
# missing dependency, malformed graph) shows up immediately in the uvicorn
# logs rather than the first time a user hits /run.
from pipeline_runner import encode_sse, stream_pipeline


# ---------------------------------------------------------------------------
# Concurrency cap.
#
# Why: Render's free tier instance is 512 MB. A single in-flight pipeline run
# peaks around 400-450 MB resident (Python imports + a pandas read of the
# 34 MB parquet inflated to ~70 MB + LLM client buffers + graph state).
# Two simultaneous runs blow past the cap and the instance is killed by the
# OOM killer mid-stream — every connected client loses their progress.
#
# An asyncio.Semaphore is the right primitive here because uvicorn runs all
# routes inside a single event loop by default. The semaphore is acquired
# inside the SSE generator so a queued request can still hold an open HTTP
# connection while it waits, and the lock is released the moment the
# generator finishes (success OR error path) thanks to the try/finally.
#
# A SECOND caller hitting /run while the slot is held will:
#   - get an immediate `event: busy` frame on its SSE stream
#   - have the connection closed
# rather than queue (which would hold an LLM-call's worth of memory in
# request buffers AND eventually OOM the box).
# ---------------------------------------------------------------------------
_PIPELINE_SLOTS = asyncio.Semaphore(1)



# ---------------------------------------------------------------------------
# FastAPI app instance.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NHTSA Project Backend",
    description=(
        "FastAPI service that hosts the LangGraph NHTSA complaint-analysis "
        "pipeline. The /run endpoint streams pipeline progress to the "
        "frontend via Server-Sent Events."
    ),
    version="0.2.0",
)


# ---------------------------------------------------------------------------
# CORS configuration.
#
# Three layers:
#   1. allow_origins   — the explicit list of trusted origins (production
#                        Vercel URL + localhost for local frontend dev).
#   2. allow_origin_regex — covers Vercel's preview-deploy URLs, which look
#                           like `https://<branch>-<project>-<team>.vercel.app`
#                           and change every push. The regex catches every
#                           *.vercel.app subdomain over HTTPS.
#   3. allow_methods / allow_headers — wildcards because the API is small
#                                       and trusted; tighten when more
#                                       endpoints exist.
#
# allow_credentials stays False because we don't ship cookies or HTTP auth
# from the browser — every request carries no per-user state on the wire.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # `next dev` local frontend
        "http://127.0.0.1:3000",   # alternate localhost spelling some browsers send
        "https://nlp-project-git-main-mataeo-ehs-projects.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    expose_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request body for POST /run.
#
# Pydantic gives us automatic 422 responses with a structured JSON error body
# when the caller sends a malformed payload (missing field, wrong type, etc.)
# — much friendlier than letting json.loads raise inside the handler.
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    user_request: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description=(
            "The natural-language request the LangGraph pipeline should "
            "process. Same content the local CLI captures from stdin or "
            "transcribes from microphone input."
        ),
    )


# ---------------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------------
@app.get("/")
async def read_root() -> dict[str, str]:
    """
    Root response. Useful as a manual sanity check that the deployment is
    live and serving the expected app version.
    """
    return {
        "service": "nhtsa-project-backend",
        "status": "ok",
        "version": app.version,
        "endpoints": "GET /health, POST /run",
    }


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """
    Lightweight liveness probe for Render's healthcheck system. Returning a
    static 200 OK is sufficient — Render only cares about the status code.
    """
    return {"status": "ok"}


@app.post("/run")
async def run(req: RunRequest) -> StreamingResponse:
    """
    Run the LangGraph NHTSA pipeline against `user_request` and stream the
    progress to the caller via Server-Sent Events (SSE).

    Why SSE rather than a single JSON response
    -------------------------------------------
    The pipeline takes 5–60 seconds end-to-end (multiple LLM calls per node,
    plus tool-calling loops). A single response would force the frontend to
    sit on a blank screen until the whole graph finishes. SSE gives us:

      - per-node progress events (`event: node_update`) the frontend renders
        as a running log so the user sees the pipeline thinking
      - a final completion event (`event: completed`) that carries the same
        `response` field the local CLI normally hands to TTS
      - structured error events (`event: error`) instead of opaque 500s when
        something blows up mid-graph

    Wire format
    -----------
    Each event is one SSE frame:

        event: <name>
        data: <json>

        event: <name>
        data: <json>

    Browser-side, the frontend can consume this with a native EventSource:

        const es = new EventSource("/run", { method: "POST", ... })  // see
        es.addEventListener("node_update", e => ...);
        es.addEventListener("completed",   e => ...);
        es.addEventListener("error",       e => ...);

    (EventSource doesn't support POST natively — the frontend will use
    fetch() with a streaming reader instead. The wire format is identical.)

    Response headers
    ----------------
    - `Content-Type: text/event-stream` — required for SSE; browsers won't
       treat the response as a stream without it.
    - `Cache-Control: no-cache` — ensures intermediate caches don't buffer
       progress events.
    - `X-Accel-Buffering: no` — tells nginx-style reverse proxies (Render's
       edge) to NOT buffer. Without it, events can sit in a buffer until
       the response closes — defeating the point of a stream.
    """
    async def event_source():
        # Try to acquire the single pipeline slot WITHOUT blocking. If another
        # request already holds it, emit one structured "busy" SSE frame so the
        # frontend can render a clean message, then close — never queue, since
        # queueing keeps memory tied up and risks OOMing the second caller's
        # eventual run too.
        if not _PIPELINE_SLOTS.locked():
            await _PIPELINE_SLOTS.acquire()
        else:
            yield encode_sse({
                "event": "busy",
                "data": {
                    "message": (
                        "The backend is already running another pipeline. "
                        "Render free-tier instances run one request at a time. "
                        "Wait for the current run to finish and try again."
                    ),
                    "retry_after_seconds": 30,
                },
            })
            return

        try:
            async for event in stream_pipeline(req.user_request):
                yield encode_sse(event)
        finally:
            # Always release — covers normal completion, exceptions inside the
            # graph, AND client disconnects (Starlette closes the generator).
            _PIPELINE_SLOTS.release()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
