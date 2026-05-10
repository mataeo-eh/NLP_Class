"""
Hosted MCP runtime for the NHTSA LangGraph backend.

Why this module exists
----------------------
The backend is server-hosted, so the model cannot rely on local code execution
for formal statistical work. This module provides a safer path:

1. Connect to a remote MCP statistics server (RMCP) over Streamable HTTP.
2. Discover a curated subset of its statistical tools at backend startup.
3. Rewrite those tool schemas so the model passes a local `dataset_id`
   instead of inlining thousands of rows into the prompt context.
4. Inject the stored dataset server-side right before the remote MCP call.

This keeps the model-facing tool contracts concise, hosted-safe, and practical
for parquet/CSV analysis workflows that would otherwise blow up context size.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

RMCP_DEFAULT_URL = "https://rmcp-server-394229601724.us-central1.run.app/mcp"
_DATASET_TTL_SECONDS = 30 * 60

try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
except Exception as exc:  # pragma: no cover - import depends on deployment env
    ClientSession = None
    streamable_http_client = None
    _MCP_IMPORT_ERROR: Exception | None = exc
else:
    _MCP_IMPORT_ERROR = None


_GLOBAL_MCP_TOOL_MANAGER: "MCPToolManager | None" = None
_STATS_DATASETS: dict[str, dict[str, Any]] = {}


def _env_flag(name: str, default: bool) -> bool:
    """Parse a conventional boolean env var."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _compact_description(text: str, *, fallback: str = "") -> str:
    """Trim third-party tool descriptions down to a concise first sentence."""
    compact = " ".join((text or fallback).split()).strip()
    if not compact:
        return fallback
    for stop in ".!?":
        idx = compact.find(stop)
        if 0 < idx <= 220:
            return compact[: idx + 1]
    if len(compact) <= 220:
        return compact
    return compact[:217].rstrip() + "..."


def _prune_expired_datasets() -> None:
    """Drop stale in-memory statistical datasets to bound hosted RAM usage."""
    now = time.time()
    expired = [
        dataset_id
        for dataset_id, record in _STATS_DATASETS.items()
        if now - float(record["updated_at"]) > _DATASET_TTL_SECONDS
    ]
    for dataset_id in expired:
        _STATS_DATASETS.pop(dataset_id, None)


def _dataset_manifest(dataset_id: str, record: dict[str, Any]) -> dict[str, Any]:
    """Return the model-facing metadata for one stored dataset."""
    columns = list(record["data"].keys())
    row_count = len(record["data"][columns[0]]) if columns else 0
    return {
        "dataset_id": dataset_id,
        "source": record["source"],
        "description": record["description"],
        "row_count": row_count,
        "columns": columns,
        "metadata": record.get("metadata", {}),
        "created_at": record["created_at"],
    }


def _normalise_dataset_columns(data: dict[str, Any]) -> dict[str, list[Any]]:
    """
    Validate column-wise dataset storage.

    Every column must be a Python list of equal length because RMCP statistical
    tools expect a table object with column_name -> [values].
    """
    if not isinstance(data, dict) or not data:
        raise ValueError("Stats datasets must be a non-empty dict of columns.")

    normalised: dict[str, list[Any]] = {}
    expected_len: int | None = None
    for column_name, values in data.items():
        if not isinstance(column_name, str) or not column_name.strip():
            raise ValueError("Every stats dataset column needs a non-empty string name.")
        if not isinstance(values, list):
            raise ValueError(f"Column '{column_name}' must be a Python list.")
        if expected_len is None:
            expected_len = len(values)
        elif len(values) != expected_len:
            raise ValueError("All stats dataset columns must have the same length.")
        normalised[column_name] = values
    return normalised


def register_stats_dataset(
    *,
    data: dict[str, Any],
    source: str,
    description: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Store a stats-ready dataset in memory and return a lightweight manifest.

    The manifest is what the model sees. The raw column arrays stay server-side
    so later RMCP tool calls can reference them by `dataset_id`.
    """
    _prune_expired_datasets()
    now = time.time()
    dataset_id = f"stats_{uuid4().hex[:12]}"
    record = {
        "data": _normalise_dataset_columns(data),
        "source": source,
        "description": description,
        "metadata": metadata or {},
        "created_at": now,
        "updated_at": now,
    }
    _STATS_DATASETS[dataset_id] = record
    return _dataset_manifest(dataset_id, record)


def get_stats_dataset(dataset_id: str) -> dict[str, Any] | None:
    """Return the raw stored dataset record, or None when missing/expired."""
    _prune_expired_datasets()
    record = _STATS_DATASETS.get(dataset_id)
    if record is None:
        return None
    record["updated_at"] = time.time()
    return record


def list_stats_dataset_manifests() -> list[dict[str, Any]]:
    """Return lightweight metadata for every currently stored stats dataset."""
    _prune_expired_datasets()
    manifests = [
        _dataset_manifest(dataset_id, record)
        for dataset_id, record in _STATS_DATASETS.items()
    ]
    manifests.sort(key=lambda item: float(item["created_at"]), reverse=True)
    return manifests


def set_global_mcp_tool_manager(manager: "MCPToolManager | None") -> None:
    """Register or clear the process-global MCP manager."""
    global _GLOBAL_MCP_TOOL_MANAGER
    _GLOBAL_MCP_TOOL_MANAGER = manager


def get_global_mcp_tool_manager() -> "MCPToolManager | None":
    """Return the process-global MCP manager, if startup created one."""
    return _GLOBAL_MCP_TOOL_MANAGER


def build_default_mcp_tool_manager() -> "MCPToolManager":
    """Create the default hosted RMCP manager from environment variables."""
    manager = MCPToolManager()
    if _env_flag("RMCP_ENABLED", True):
        manager.add_streamable_http_server(
            "rmcp",
            os.getenv("RMCP_SERVER_URL", RMCP_DEFAULT_URL),
        )
    return manager


def convert_mcp_tool_to_openai(mcp_tool) -> dict:
    """Convert one discovered MCP tool into an OpenAI-style function schema."""
    schema = getattr(mcp_tool, "inputSchema", {})
    if not isinstance(schema, dict):
        schema = {"type": "object", "properties": {}, "required": []}

    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": _compact_description(
                getattr(mcp_tool, "description", ""),
                fallback=f"Call remote MCP tool '{mcp_tool.name}'.",
            ),
            "parameters": schema,
        },
    }


def _wrap_dataset_tool_schema(mcp_tool) -> dict | None:
    """
    Rewrite an RMCP tool schema so the model passes `dataset_id` instead of `data`.

    We expose only data-driven statistical tools. File-read/write tools are
    intentionally skipped because the backend already owns the complaint data
    locally and should not hand the model an arbitrary remote filesystem surface.
    """
    schema = getattr(mcp_tool, "inputSchema", {})
    if not isinstance(schema, dict):
        return None

    properties = dict(schema.get("properties") or {})
    if "data" not in properties:
        return None
    if mcp_tool.name.startswith("read_") or mcp_tool.name.startswith("write_"):
        return None

    required = [name for name in (schema.get("required") or []) if name != "data"]
    wrapped_properties = {
        "dataset_id": {
            "type": "string",
            "description": (
                "Dataset id produced by build_complaint_stats_dataset or "
                "build_csv_parquet_label_stats_dataset."
            ),
        }
    }
    for name, value in properties.items():
        if name == "data":
            continue
        wrapped_properties[name] = value

    wrapped_required = ["dataset_id", *required]
    wrapped_schema = {
        "type": "object",
        "properties": wrapped_properties,
        "required": wrapped_required,
        "additionalProperties": schema.get("additionalProperties", False),
    }
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": _compact_description(
                getattr(mcp_tool, "description", ""),
                fallback=f"Run the remote statistical tool '{mcp_tool.name}'.",
            ),
            "parameters": wrapped_schema,
        },
    }


class MCPToolManager:
    """
    Manage long-lived remote MCP sessions for the hosted backend.

    The manager performs discovery at startup, caches the wrapped tool schemas
    that should be exposed to the model, and executes remote tool calls on the
    backend event loop when LangGraph asks for them from worker threads.
    """

    def __init__(self, server_configs: Optional[Dict[str, dict[str, Any]]] = None):
        self.server_configs = server_configs or {}
        self.sessions: dict[str, Any] = {}
        self._session_stacks: dict[str, AsyncExitStack] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._tool_registry: dict[str, dict[str, Any]] = {}
        self._startup_errors: dict[str, str] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def add_streamable_http_server(self, name: str, url: str, *, enabled: bool = True) -> None:
        """Register one remote Streamable HTTP MCP server."""
        self.server_configs[name] = {"url": url, "enabled": enabled}

    async def startup(self, *, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Connect to every enabled MCP server and cache safe wrapped tools."""
        self._loop = loop or asyncio.get_running_loop()
        self._startup_errors.clear()
        self._tool_registry.clear()

        if _MCP_IMPORT_ERROR is not None:
            self._startup_errors["mcp_import"] = str(_MCP_IMPORT_ERROR)
            logger.warning("MCP SDK import failed: %s", _MCP_IMPORT_ERROR)
            return

        for server_name, config in self.server_configs.items():
            if not config.get("enabled", True):
                continue

            stack = AsyncExitStack()
            try:
                read_stream, write_stream, _ = await stack.enter_async_context(
                    streamable_http_client(config["url"])
                )
                session = await stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                tools_result = await session.list_tools()

                self.sessions[server_name] = session
                self._session_stacks[server_name] = stack
                self._locks[server_name] = asyncio.Lock()

                for remote_tool in tools_result.tools:
                    wrapped = _wrap_dataset_tool_schema(remote_tool)
                    if wrapped is None:
                        continue
                    self._tool_registry[remote_tool.name] = {
                        "server_name": server_name,
                        "remote_tool": remote_tool,
                        "wrapped_tool": wrapped,
                    }
            except Exception as exc:  # pragma: no cover - depends on live remote
                self._startup_errors[server_name] = str(exc)
                logger.warning(
                    "Failed to initialise remote MCP server '%s' at %s: %s",
                    server_name,
                    config.get("url"),
                    exc,
                )
                await stack.aclose()

    async def shutdown(self) -> None:
        """Close every long-lived MCP session cleanly."""
        for stack in self._session_stacks.values():
            try:
                await stack.aclose()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.warning("Failed to close MCP session cleanly: %s", exc)
        self.sessions.clear()
        self._session_stacks.clear()
        self._locks.clear()
        self._tool_registry.clear()

    def get_wrapped_openai_tools(self) -> List[dict]:
        """Return the safe model-facing RMCP tool schemas discovered at startup."""
        return [
            payload["wrapped_tool"]
            for _, payload in sorted(self._tool_registry.items(), key=lambda item: item[0])
        ]

    def get_wrapped_tool_names(self) -> set[str]:
        """Return the names of all safe wrapped RMCP tools currently available."""
        return set(self._tool_registry.keys())

    def has_wrapped_tool(self, tool_name: str) -> bool:
        """True when the named wrapped RMCP tool is available for execution."""
        return tool_name in self._tool_registry

    def describe_status(self) -> dict[str, Any]:
        """Return a JSON-friendly health snapshot for the hosted statistics layer."""
        return {
            "configured_servers": sorted(self.server_configs.keys()),
            "connected_servers": sorted(self.sessions.keys()),
            "wrapped_tool_count": len(self._tool_registry),
            "wrapped_tools": sorted(self._tool_registry.keys()),
            "startup_errors": dict(self._startup_errors),
        }

    async def execute_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """
        Execute one wrapped RMCP tool by resolving its dataset id server-side.

        The model never passes raw data arrays back through prompt context. It
        passes `dataset_id`, and the backend expands that into the original
        column-wise table just before the remote call.
        """
        payload = self._tool_registry.get(tool_name)
        if payload is None:
            raise ValueError(f"Wrapped RMCP tool '{tool_name}' is not available.")

        args = dict(tool_args or {})
        dataset_id = str(args.pop("dataset_id", "")).strip()
        if not dataset_id:
            raise ValueError("RMCP statistical tools require a non-empty dataset_id.")

        dataset_record = get_stats_dataset(dataset_id)
        if dataset_record is None:
            raise ValueError(
                f"Stats dataset '{dataset_id}' was not found or has expired."
            )

        remote_args = {"data": dataset_record["data"], **args}
        server_name = payload["server_name"]
        session = self.sessions.get(server_name)
        if session is None:
            raise ValueError(f"Remote MCP server '{server_name}' is not connected.")

        lock = self._locks[server_name]
        async with lock:
            result = await session.call_tool(tool_name, arguments=remote_args)
        return self._format_result(result)

    def execute_tool_call_sync(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Thread-safe bridge from LangGraph worker threads to the async MCP session."""
        if self._loop is None:
            raise RuntimeError("MCP tool manager has not been started.")
        future = asyncio.run_coroutine_threadsafe(
            self.execute_tool_call(tool_name, tool_args),
            self._loop,
        )
        return future.result()

    def _format_result(self, result: Any) -> str:
        """Normalise MCP tool results into one string for ToolMessage content."""
        structured = getattr(result, "structuredContent", None)
        if structured not in (None, {}):
            return json.dumps(structured, indent=2, default=str)

        content = getattr(result, "content", result)
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                parts.append(text if text is not None else str(item))
            return "\n".join(parts)
        return str(content)


@tool
def rmcp_status() -> str:
    """Report whether the hosted RMCP statistical backend is currently available."""
    manager = get_global_mcp_tool_manager()
    status = (
        manager.describe_status()
        if manager is not None
        else {
            "configured_servers": [],
            "connected_servers": [],
            "wrapped_tool_count": 0,
            "wrapped_tools": [],
            "startup_errors": {"manager": "RMCP manager has not been initialised."},
        }
    )
    return json.dumps(status, indent=2)


@tool
def list_stats_datasets() -> str:
    """List in-memory stats datasets that can be reused by RMCP statistical tools."""
    payload = {"datasets": list_stats_dataset_manifests()}
    return json.dumps(payload, indent=2)


@tool
def describe_stats_dataset(dataset_id: str) -> str:
    """Describe one stored stats dataset without returning its raw rows."""
    dataset_id = str(dataset_id).strip()
    if not dataset_id:
        return json.dumps({"error": "dataset_id must be a non-empty string."}, indent=2)
    record = get_stats_dataset(dataset_id)
    if record is None:
        return json.dumps(
            {"error": f"Stats dataset '{dataset_id}' was not found or has expired."},
            indent=2,
        )
    return json.dumps(_dataset_manifest(dataset_id, record), indent=2)
