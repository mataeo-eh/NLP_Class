"""
NHTSA Project — FastAPI backend.

What this file does
-------------------
This is the ASGI entrypoint that uvicorn imports as `main:app` on Railway.
It exposes the LangGraph NHTSA complaint-analysis pipeline as an HTTP API:

    GET  /             -> service banner (sanity check)
    GET  /health       -> 200 OK probe for Railway's healthcheck
    POST /session/warmup -> returns the CLI greeting text for frontend warmup
    GET  /audio/tts/options -> provider + voice registry for the frontend
    POST /run          -> Server-Sent Events stream of pipeline progress
    POST /audio/transcribe -> browser audio upload to transcript JSON
    POST /audio/speech     -> text to browser-playable WAV bytes
    POST /audio/speech/stream -> low-latency PCM byte stream for browser playback

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
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

# pipeline_runner sets the headless flag and imports the compiled graph at
# module load. Importing it here means any startup-time failure (bad path,
# missing dependency, malformed graph) shows up immediately in the uvicorn
# logs rather than the first time a user hits /run.
from pipeline_runner import WELCOME_TTS_TEXT, encode_sse, stream_pipeline
from LLM_Tools.MCP_To_Tools import (
    build_default_mcp_tool_manager,
    set_global_mcp_tool_manager,
)
from audio_io import (
    MAX_AUDIO_UPLOAD_BYTES,
    get_streaming_audio_contract,
    synthesize_speech_pcm_stream,
    synthesize_speech_wav,
    transcribe_audio_bytes,
)
from tts_registry import (
    get_default_hosted_tts_preview_text,
    get_default_hosted_tts_provider,
    get_default_hosted_voice_preset,
    list_hosted_tts_providers,
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
_AGENTIC_SESSION_TTL_SECONDS = 30 * 60
_AGENTIC_SESSIONS: dict[str, dict[str, Any]] = {}


def _prune_expired_agentic_sessions() -> None:
    """
    Drop stale in-memory agentic sessions so hosted follow-up memory is bounded.

    The frontend can explicitly start a new conversation at any time, but users
    can also abandon tabs. A short TTL keeps those abandoned transcripts from
    living forever on the backend while still leaving enough time for normal
    conversational follow-ups.
    """
    now = time.time()
    expired_session_ids = [
        session_id
        for session_id, record in _AGENTIC_SESSIONS.items()
        if now - float(record["updated_at"]) > _AGENTIC_SESSION_TTL_SECONDS
    ]
    for session_id in expired_session_ids:
        _AGENTIC_SESSIONS.pop(session_id, None)


def _load_agentic_session(session_id: str) -> dict[str, Any] | None:
    _prune_expired_agentic_sessions()
    record = _AGENTIC_SESSIONS.get(session_id)
    if record is None:
        return None
    return dict(record["state"])


def _store_agentic_session(session_id: str, state: dict[str, Any]) -> None:
    _AGENTIC_SESSIONS[session_id] = {
        "state": dict(state),
        "updated_at": time.time(),
    }


def _clear_agentic_session(session_id: str) -> None:
    _AGENTIC_SESSIONS.pop(session_id, None)


def _state_is_resumable_agentic(state: dict[str, Any]) -> bool:
    """
    Persist only sessions that can actually continue a later follow-up turn.

    The hosted continuation contract applies only to the agentic branch, and it
    requires a serialized internal message transcript. Non-agentic runs or
    agentic runs that never produced conversation_messages should not occupy the
    session store.
    """
    return (
        state.get("task_type") == "agentic_retrieve_and_analyze"
        and state.get("agentic_subtype") in {"agentic_analyze", "agentic_explore"}
        and isinstance(state.get("conversation_messages"), list)
        and len(state.get("conversation_messages") or []) > 0
    )



# ---------------------------------------------------------------------------
# FastAPI lifespan.
#
# The hosted RMCP statistical backend is a shared async resource. FastAPI's
# lifespan hook is the right place to create the long-lived MCP session once at
# startup and tear it down once at shutdown.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    mcp_manager = build_default_mcp_tool_manager()
    set_global_mcp_tool_manager(mcp_manager)
    try:
        await mcp_manager.startup(loop=asyncio.get_running_loop())
        app.state.rmcp_status = mcp_manager.describe_status()
        yield
    finally:
        app.state.rmcp_status = {
            "configured_servers": [],
            "connected_servers": [],
            "wrapped_tool_count": 0,
            "wrapped_tools": [],
            "startup_errors": {"shutdown": "RMCP manager is offline."},
        }
        try:
            await mcp_manager.shutdown()
        finally:
            set_global_mcp_tool_manager(None)


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
    version="0.4.0",
    lifespan=lifespan,
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
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description=(
            "Opaque browser-generated conversation identifier. Follow-up turns "
            "reuse the same id so the backend can restore the prior agentic "
            "message/tool transcript."
        ),
    )
    continue_session: bool = Field(
        default=False,
        description=(
            "True when this request is a follow-up inside an already-running "
            "agentic conversation and should reuse the stored session state "
            "instead of starting from a blank pipeline state."
        ),
    )


class SpeechRequest(BaseModel):
    """
    Hosted speech request accepted by both /audio/speech and /audio/speech/stream.

    The frontend can send a provider/voice pair on every request so hosted users
    are not forced to share one process-wide TTS selector. That also makes
    mid-run switching feasible: future narration requests can move to a new
    provider/voice without restarting the pipeline.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Text to synthesize as browser-playable WAV audio.",
    )
    tts_provider: str | None = Field(
        default=None,
        description=(
            "Optional hosted TTS provider id. When omitted, the backend keeps "
            "the current hosted default provider."
        ),
    )
    voice_preset: str | None = Field(
        default=None,
        description=(
            "Optional voice preset for the selected provider. When omitted, the "
            "provider's default hosted voice is used."
        ),
    )


class TtsVoiceOption(BaseModel):
    """One selectable hosted voice for one provider."""

    id: str
    label: str
    description: str


class TtsProviderOption(BaseModel):
    """Hosted provider metadata plus its provider-specific voice list."""

    id: str
    label: str
    description: str
    supports_streaming: bool
    default_voice_preset: str
    voices: list[TtsVoiceOption]


class TtsOptionsResponse(BaseModel):
    """Frontend bootstrap payload for provider and voice dropdowns."""

    default_tts_provider: str
    default_voice_preset: str
    default_preview_text: str
    providers: list[TtsProviderOption]


class WarmupResponse(BaseModel):
    """
    Hosted warmup payload for the minimal web client.

    The frontend uses the same greeting copy as the local CLI so the hosted
    flow begins with the familiar spoken handoff, then transitions into either
    typed input or push-to-talk upload.
    """

    welcome_text: str = Field(
        ...,
        description="Greeting text the frontend should speak through /audio/speech/stream.",
    )
    ready_prompt: str = Field(
        ...,
        description="Short UI status message to show when the browser can capture or send input.",
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
        "endpoints": (
            "GET /health, POST /session/warmup, GET /audio/tts/options, POST /run, "
            "POST /audio/transcribe, POST /audio/speech, POST /audio/speech/stream"
        ),
    }


@app.get("/health")
async def healthcheck() -> dict[str, Any]:
    """
    Lightweight liveness probe for Railway's healthcheck system. Returning a
    static 200 OK is sufficient — Railway only cares about the status code.
    """
    return {
        "status": "ok",
        "rmcp": getattr(
            app.state,
            "rmcp_status",
            {
                "configured_servers": [],
                "connected_servers": [],
                "wrapped_tool_count": 0,
                "wrapped_tools": [],
                "startup_errors": {"manager": "RMCP status has not been initialised."},
            },
        ),
    }


@app.post("/session/warmup", response_model=WarmupResponse)
async def warmup_session() -> WarmupResponse:
    """
    Return the hosted warmup text that mirrors the CLI greeting.

    The route is intentionally lightweight: it does not open any audio devices
    or create server-side session state. The browser remains responsible for
    microphone permission and push-to-talk capture, while the backend stays the
    source of truth for the greeting copy and ready-state prompt.
    """
    return WarmupResponse(
        welcome_text=WELCOME_TTS_TEXT,
        ready_prompt="Speak now.",
    )


@app.get("/audio/tts/options", response_model=TtsOptionsResponse)
async def get_tts_options() -> TtsOptionsResponse:
    """
    Return the hosted TTS provider and voice registry for the frontend.

    The response is explicit rather than inferred from backend internals so the
    client can deterministically:

    * populate the provider dropdown,
    * swap the voice dropdown options when the provider changes,
    * preserve the current hosted default selection,
    * decide whether the selected provider supports the low-latency stream route.
    """
    default_provider = get_default_hosted_tts_provider()
    return TtsOptionsResponse(
        default_tts_provider=default_provider,
        default_voice_preset=get_default_hosted_voice_preset(default_provider),
        default_preview_text=get_default_hosted_tts_preview_text(),
        providers=[
            TtsProviderOption.model_validate(provider)
            for provider in list_hosted_tts_providers()
        ],
    )


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
                tts_provider=req.tts_provider,
                voice_preset=req.voice_preset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            status = 503 if "not configured" in str(exc) else 502
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except Exception as exc:
            # Keep third-party SDK/runtime failures inside the normal FastAPI
            # response path. If an exception escapes this route entirely,
            # Railway returns a plain 500 without CORS headers and browsers
            # report only "Load failed", hiding the actionable error.
            raise HTTPException(status_code=502, detail=str(exc)) from exc

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


@app.post("/audio/speech/stream")
async def stream_speech(req: SpeechRequest) -> StreamingResponse:
    """
    Stream raw PCM bytes for immediate browser playback.

    Why this route exists
    ---------------------
    `/audio/speech` intentionally returns a finished WAV file, which is useful
    for replay controls but forces the client to wait for the full synthesis to
    finish before playback can begin. This companion route exposes Deepgram's
    chunked output as a streaming HTTP body so the frontend can schedule each
    PCM chunk in the Web Audio API as soon as it arrives.

    Provider note
    -------------
    Deepgram, Cartesia, and OpenAI are wired into this raw-audio contract
    today, but they do NOT share the same byte format:

    * Deepgram -> linear16, 24000 Hz, mono
    * Cartesia -> pcm_f32le, 44100 Hz, mono
    * OpenAI   -> pcm16, 24000 Hz, mono (best-effort inferred contract)

    The registry route tells the frontend which providers support streaming.
    Unsupported providers should use the buffered WAV endpoint instead of
    guessing.

    Explicit frontend contract
    --------------------------
    The response body is NOT a self-describing audio container. The frontend
    must read the headers below and handle them programmatically:

    * `X-Audio-Codec: <provider-specific raw codec>`
    * `X-Audio-Sample-Rate: <provider-specific sample rate>`
    * `X-Audio-Channels: 1`

    If the browser client cannot handle that contract, it should fall back to
    the buffered WAV endpoint instead of guessing.
    """
    if _AUDIO_SLOTS.locked():
        raise HTTPException(
            status_code=429,
            detail="Another audio request is already running. Try again when it finishes.",
        )

    await _AUDIO_SLOTS.acquire()
    try:
        try:
            stream_contract = get_streaming_audio_contract(
                tts_provider=req.tts_provider,
                voice_preset=req.voice_preset,
            )
            pcm_chunks = synthesize_speech_pcm_stream(
                req.text,
                tts_provider=req.tts_provider,
                voice_preset=req.voice_preset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            status = 503 if "not configured" in str(exc) else 502
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        def generate_pcm_chunks():
            try:
                yield from pcm_chunks
            finally:
                _AUDIO_SLOTS.release()

        return StreamingResponse(
            generate_pcm_chunks(),
            media_type="application/octet-stream",
            headers={
                "Cache-Control": "no-store",
                "X-Audio-Codec": str(stream_contract["codec"]),
                "X-Audio-Sample-Rate": str(stream_contract["sample_rate"]),
                "X-Audio-Channels": str(stream_contract["channels"]),
            },
        )
    except Exception:
        _AUDIO_SLOTS.release()
        raise


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
    session_id = (req.session_id or "").strip() or str(uuid4())
    if not req.continue_session:
        # A non-follow-up request is a deliberate reset boundary for this
        # browser conversation id. Clearing now prevents an accidental reuse of
        # stale agentic history if the frontend chooses to recycle the same id.
        _clear_agentic_session(session_id)

    prior_state = None
    if req.continue_session:
        prior_state = _load_agentic_session(session_id)
        if prior_state is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "The requested agentic session expired or was never created. "
                    "Start a new conversation before sending a follow-up."
                ),
            )

    final_state_box: dict[str, Any] = {}

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
            async for event in stream_pipeline(
                req.user_request,
                prior_state=prior_state,
                final_state_sink=final_state_box,
            ):
                if event.get("event") in {"started", "completed"}:
                    event = {
                        **event,
                        "data": {
                            **dict(event.get("data") or {}),
                            "session_id": session_id,
                        },
                    }
                yield encode_sse(event)
        finally:
            final_state = final_state_box.get("state")
            succeeded = final_state_box.get("succeeded") is True
            if succeeded and isinstance(final_state, dict):
                if _state_is_resumable_agentic(final_state):
                    _store_agentic_session(session_id, final_state)
                else:
                    _clear_agentic_session(session_id)
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
