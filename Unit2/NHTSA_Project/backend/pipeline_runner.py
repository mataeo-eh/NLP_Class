"""
pipeline_runner.py
------------------
Adapter that bridges the LangGraph NHTSA pipeline (built for an interactive
CLI with audio + stdin) into the FastAPI backend (a headless web service with
no microphone, no speakers, and no human at stdin).

Responsibilities
----------------
1. Fix up sys.path so the existing LangGraph modules — which use bare imports
   like `from State import State` and `from Nodes import ...` — resolve when
   uvicorn is launched from this `backend/` directory.

2. Flip the process-wide runtime flags BEFORE the graph module is imported:
       set_audio_enabled(False)   - no mic/speakers exist on Render
       set_headless_mode(True)    - tells voice_ask_user / code_exec /
                                    confirm_csv_write / narrate_* to refuse
                                    or no-op instead of blocking on input()
   The order matters: the graph module's transitive imports touch tools that
   read these flags at call time, so they must be set first.

3. Expose `stream_pipeline(user_request)` — an async generator that yields
   plain dicts shaped like SSE events. The FastAPI route in main.py is
   responsible for formatting those dicts into the on-the-wire SSE protocol.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections import deque
from pathlib import Path
from typing import Any, AsyncIterator


# ---------------------------------------------------------------------------
# 1. sys.path bootstrap.
#
# This file lives at:  <NHTSA_Project>/backend/pipeline_runner.py
# The LangGraph code expects the following layout to be importable:
#   <NHTSA_Project>/             <- parent; holds Prompts.py, NHTSA/, etc.
#   <NHTSA_Project>/LangGraph/   <- bare imports: State, Nodes, Edges, config, Graph
#   <NHTSA_Project>/LLM_Tools/   <- LLM_Tools.NHTSA_Query_Tools subpackage
#   <NHTSA_Project>/Project_Tools/  <- Audio_Playback, Voice_Tools, etc.
#
# Insert at index 0 so these paths shadow any same-named globals on the
# Python path (e.g. there is also a top-level `Graph.py` library on PyPI
# called `pygraphviz/graphviz`; insert order keeps that out of the way).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANGGRAPH_DIR = PROJECT_ROOT / "LangGraph"
LLM_TOOLS_DIR = PROJECT_ROOT / "LLM_Tools"

for path in (str(LANGGRAPH_DIR), str(LLM_TOOLS_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# 2. Flip runtime flags BEFORE importing the graph.
#
# Project_Tools.Runtime_Options uses module-level globals as the source of
# truth for these flags, so the import chain that follows can read the right
# values immediately — even if a node module caches `is_headless_mode` at
# import time (none currently does, but this keeps that future-safe).
# ---------------------------------------------------------------------------
from Project_Tools.Runtime_Options import (  # noqa: E402  (intentional import order)
    set_audio_enabled,
    set_headless_narration_sink,
    set_headless_mode,
)

set_audio_enabled(False)
set_headless_mode(True)


# ---------------------------------------------------------------------------
# 3. Import the compiled graph.
#
# Graph.py only runs its CLI under `if __name__ == "__main__":` so importing
# the module is side-effect free apart from compiling the StateGraph (which
# is what we want).  `graph_app` is the `langgraph.graph.state.CompiledStateGraph`
# object exposed by Graph.py at module level via `app = graph.compile()`.
# ---------------------------------------------------------------------------
import Graph as _graph_module  # noqa: E402

graph_app = _graph_module.app
WELCOME_TTS_TEXT = getattr(
    _graph_module,
    "WELCOME_TTS_TEXT",
    "Welcome back! What NHTSA adventure shall we embark on?",
)


# ---------------------------------------------------------------------------
# State seed.
#
# Mirrors the dict that Graph.py's main() passes into `app.invoke(...)`. Every
# field declared in LangGraph/State.py needs an initial value of the right
# type so nodes that read defaults (e.g. iteration_log.append) work without
# a KeyError.
# ---------------------------------------------------------------------------
def _initial_state(user_request: str) -> dict[str, Any]:
    return {
        "user_request": user_request,
        "task_type": "",
        "query_result": [],          # list[dict] — populated by retrieve_data
        "reasoning_output": "",
        "analysis": [],              # list[dict] — populated by analyze
        "analysis_prompt_name": "",
        "response": "",
        "csv_write_confirmed": False,
        "agentic_subtype": "",
        "iteration_log": [],
    }


# ---------------------------------------------------------------------------
# JSON safety.
#
# State deltas can carry pandas/numpy values (NaN floats, np.int64, etc.) that
# json.dumps refuses by default. We do a single recursive normalisation pass
# and fall back to str() for anything unrecognised — losing structured access
# to that field is fine since the frontend renders deltas as informational
# progress, not as load-bearing data.
# ---------------------------------------------------------------------------
def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        # Guard against NaN / Infinity, which json.dumps emits as the literal
        # tokens NaN / Infinity — invalid JSON for browsers' EventSource.
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    # numpy / pandas scalars expose .item() to convert to a native Python type.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except Exception:
            pass
    # Last resort — render whatever it is as a string so the SSE event still
    # serialises cleanly even for exotic objects.
    return str(value)


def _summarise_delta(delta: dict[str, Any]) -> dict[str, Any]:
    """
    Trim a node's state delta down to a payload that's safe and useful to send
    over the wire. Long fields (full row dumps, full message histories) become
    counts so the SSE event stays small; short scalar fields are passed through.
    """
    summary: dict[str, Any] = {}
    for key, value in delta.items():
        if key == "response":
            # The terminal user-facing prose is emitted exactly once via the
            # dedicated `completed` event below. Excluding it from per-node
            # updates prevents terminal nodes (retrieve_data, agentic_* paths)
            # from sending the same final answer twice back-to-back.
            continue
        if key in ("query_result", "analysis", "iteration_log") and isinstance(value, list):
            # Send a count plus the first element as a sample — enough for the
            # frontend to display "retrieved 12 rows" without shipping the
            # entire dataset over the SSE channel for every node update.
            summary[key] = {
                "count": len(value),
                "sample": _to_jsonable(value[0]) if value else None,
            }
        else:
            summary[key] = _to_jsonable(value)
    return summary


def _drain_narration_events(queue: deque[str]) -> list[dict[str, Any]]:
    """Convert queued Mercury narration strings into explicit SSE events."""
    events: list[dict[str, Any]] = []
    while queue:
        text = queue.popleft().strip()
        if not text:
            continue
        events.append(
            {
                "event": "narration",
                "data": {
                    "channel": "progress",
                    "text": text,
                },
            }
        )
    return events


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
async def stream_pipeline(user_request: str) -> AsyncIterator[dict[str, Any]]:
    """
    Run the LangGraph pipeline on the given user_request and yield SSE-shaped
    progress events.

    Event shape (matches the JSON the FastAPI route encodes onto the wire):

        {"event": "started",     "data": {"user_request": "..."}}
        {"event": "node_update", "data": {"node": "<name>", "delta": {...}}}
        {"event": "completed",   "data": {"final": {...}}}
        {"event": "error",       "data": {"message": "...", "type": "..."}}

    The frontend can use the `event` field as the SSE event name (browsers
    subscribe to specific event names via `EventSource.addEventListener`).
    """
    yield {"event": "started", "data": {"user_request": user_request}}

    final_state: dict[str, Any] = {}
    narration_queue: deque[str] = deque()

    def enqueue_narration(text: str) -> None:
        narration_queue.append(text)

    set_headless_narration_sink(enqueue_narration)
    try:
        # LangGraph's astream() with default stream_mode emits one chunk per
        # node completion. Each chunk is a {node_name: state_delta} dict. We
        # forward each node->delta pair as its own SSE event so the frontend
        # can render incremental progress.
        async for chunk in graph_app.astream(_initial_state(user_request)):
            if not isinstance(chunk, dict):
                # Defensive: if a future LangGraph version changes the chunk
                # shape, fall back to a generic event rather than crashing.
                yield {"event": "node_update", "data": {"raw": _to_jsonable(chunk)}}
                for narration_event in _drain_narration_events(narration_queue):
                    yield narration_event
                continue

            for node_name, delta in chunk.items():
                summary = _summarise_delta(delta) if isinstance(delta, dict) else _to_jsonable(delta)
                # Track the latest known full state so we can surface a final
                # response field even though astream only returns deltas.
                if isinstance(delta, dict):
                    final_state.update(delta)
                # If the node only contributed the terminal `response` field,
                # `_summarise_delta()` returns {} on purpose. In that case the
                # very next event will be `completed` with the same response, so
                # suppress the redundant intermediate node_update entirely.
                if summary:
                    yield {
                        "event": "node_update",
                        "data": {"node": str(node_name), "delta": summary},
                    }
                for narration_event in _drain_narration_events(narration_queue):
                    yield narration_event

        for narration_event in _drain_narration_events(narration_queue):
            yield narration_event
        yield {
            "event": "completed",
            "data": {
                # `response` is the field the original CLI feeds into TTS via
                # json_to_spoken_text. The frontend can render it directly as
                # the final answer text.
                "response": _to_jsonable(final_state.get("response", "")),
                "task_type": final_state.get("task_type", ""),
                "analysis_prompt_name": final_state.get("analysis_prompt_name", ""),
                "row_count": len(final_state.get("query_result") or []),
                "analysis_count": len(final_state.get("analysis") or []),
            },
        }
    except Exception as exc:
        # Surface errors to the frontend rather than letting the SSE stream die
        # silently. Keep the traceback server-side only — never ship internal
        # stack traces over the wire because they leak file paths and library
        # versions to whoever can hit /run.
        traceback.print_exc()
        for narration_event in _drain_narration_events(narration_queue):
            yield narration_event
        yield {
            "event": "error",
            "data": {
                "message": str(exc),
                "type": type(exc).__name__,
            },
        }
    finally:
        set_headless_narration_sink(None)


def encode_sse(event: dict[str, Any]) -> str:
    """
    Format one event dict as an SSE wire frame.

    SSE protocol (per the WHATWG HTML spec):
      - One field per line, in `field: value` form.
      - The standard fields are `event`, `data`, `id`, and `retry`.
      - `data` may span multiple lines if the value contains newlines; each
        physical line in the value is emitted as its own `data:` line.
      - A blank line terminates the event.

    We always JSON-encode the data payload so the frontend can `JSON.parse`
    `e.data` uniformly across event types.
    """
    name = event.get("event", "message")
    payload = json.dumps(event.get("data", {}), ensure_ascii=False)
    # `data:` cannot contain raw newlines — split and re-prefix each line.
    data_lines = "\n".join(f"data: {line}" for line in payload.split("\n"))
    return f"event: {name}\n{data_lines}\n\n"
