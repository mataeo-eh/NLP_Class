"""
NHTSA Project — FastAPI backend.

What this file does
-------------------
This is the ASGI entrypoint that uvicorn imports as `main:app` on Railway.
It exposes the LangGraph NHTSA complaint-analysis pipeline as an HTTP API:

    GET  /             -> service banner (sanity check)
    GET  /health       -> 200 OK probe for Railway's healthcheck
    POST /run          -> Server-Sent Events stream of pipeline progress
    POST /audio/transcribe -> browser audio upload to transcript JSON
    POST /audio/speech     -> text to browser-playable WAV bytes

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
different origin (this backend URL) unless the backend server explicitly
opts in via CORS response headers.

CORS is a SERVER concern, not a client concern: Vercel does not need to
configure anything; this FastAPI app does. The `allow_origins` list below
is the allowlist of caller origins this server will accept, NOT a list of
where this server can be hosted. Adding the backend URL itself to that list
would do nothing useful, since browsers stamp the request's `Origin` with
the *frontend's* domain.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

# pipeline_runner sets the headless flag and imports the compiled graph at
# module load. Importing it here means any startup-time failure (bad path,
# missing dependency, malformed graph) shows up immediately in the uvicorn
# logs rather than the first time a user hits /run.
from pipeline_runner import encode_sse, stream_pipeline
from audio_io import (
    MAX_AUDIO_UPLOAD_BYTES,
    synthesize_speech_wav,
    transcribe_audio_bytes,
)


# ---------------------------------------------------------------------------
# Concurrency cap.
#
# Why: smaller hosted instances have finite memory. A single in-flight pipeline run
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
_AUDIO_SLOTS = asyncio.Semaphore(1)



# ---------------------------------------------------------------------------
# FastAPI app instance.
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NHTSA Project Backend",
    description=(
        "FastAPI service that hosts the LangGraph NHTSA complaint-analysis "
        "pipeline. The /run endpoint streams pipeline progress to the "
        "frontend via Server-Sent Events. The /audio/* endpoints provide "
        "browser-native speech-to-text and text-to-speech I/O."
    ),
    version="0.3.0",
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


class SpeechRequest(BaseModel):
    """
    Request body for POST /audio/speech.

    Deepgram's speech endpoint accepts several synthesis fields. The hosted backend
    exposes only the ones this app needs:

    * text: final answer or any frontend text to speak.
    * speed: positive speech-rate multiplier forwarded to Deepgram.

    The Deepgram model id is intentionally owned server-side by the existing
    Runtime_Options voice preset. For Deepgram, that model id is also the voice
    selection, so the frontend cannot accidentally bypass the pre-selected
    voice.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Text to synthesize as browser-playable WAV audio.",
    )
    speed: float = Field(
        default=1.0,
        gt=0,
        le=2.0,
        description="Speech speed multiplier forwarded to Deepgram.",
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
        "endpoints": "GET /health, POST /run, POST /audio/transcribe, POST /audio/speech",
    }


@app.get("/health")
async def healthcheck() -> dict[str, str]:
    """
    Lightweight liveness probe for Railway's healthcheck system. Returning a
    static 200 OK is sufficient — Railway only cares about the status code.
    """
    return {"status": "ok"}


@app.post("/audio/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(
        ...,
        description=(
            "Browser-recorded or user-selected audio file. Supported containers "
            "include webm, wav, mp3, m4a/mp4, ogg/opus, flac, and aac."
        ),
    ),
    language: str = "en",
) -> dict:
    """
    Convert uploaded browser audio into text for POST /run.

    FastAPI exposes UploadFile with three concrete fields we consume here:
    `filename`, `content_type`, and the async `read()` method. The body is read
    up to MAX_AUDIO_UPLOAD_BYTES + 1 so oversized uploads are rejected before
    bytes are forwarded to the hosted transcription API.
    """
    if _AUDIO_SLOTS.locked():
        raise HTTPException(
            status_code=429,
            detail="Another audio request is already running. Try again when it finishes.",
        )

    await _AUDIO_SLOTS.acquire()
    try:
        payload = await audio.read(MAX_AUDIO_UPLOAD_BYTES + 1)
        try:
            return await asyncio.to_thread(
                transcribe_audio_bytes,
                payload,
                audio.filename,
                audio.content_type,
                language,
            )
        except ValueError as exc:
            detail = str(exc)
            if "limit is" in detail:
                status_code = 413
            elif "content type" in detail:
                status_code = 415
            else:
                status_code = 400
            raise HTTPException(status_code=status_code, detail=detail) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _AUDIO_SLOTS.release()


@app.post("/audio/speech")
async def synthesize_speech(req: SpeechRequest) -> Response:
    """
    Convert text into browser-playable WAV bytes.

    The CLI TTS helpers play audio locally. This route deliberately returns an
    HTTP `audio/wav` body instead, which lets the frontend attach the bytes to an
    `<audio>` element or Web Audio API pipeline without server-side speakers.
    """
    if _AUDIO_SLOTS.locked():
        raise HTTPException(
            status_code=429,
            detail="Another audio request is already running. Try again when it finishes.",
        )

    await _AUDIO_SLOTS.acquire()
    try:
        try:
            audio_bytes = await asyncio.to_thread(
                synthesize_speech_wav,
                req.text,
                speed=req.speed,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            status = 503 if "not configured" in str(exc) else 502
            raise HTTPException(status_code=status, detail=str(exc)) from exc

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'inline; filename="nhtsa-response.wav"',
                "Cache-Control": "no-store",
            },
        )
    finally:
        _AUDIO_SLOTS.release()


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
    - `X-Accel-Buffering: no` — tells nginx-style reverse proxies (Railway's
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
                        "This deployment runs one pipeline request at a time. "
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
