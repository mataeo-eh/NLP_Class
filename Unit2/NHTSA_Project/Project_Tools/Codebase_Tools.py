"""
Codebase_Tools.py
-----------------
Read-only LangChain-compatible tools that let the LLM navigate and inspect
the NHTSA_Project source tree without leaving the project boundary.

Both tools are scoped to PROJECT_ROOT (NHTSA_Project/) — any path that resolves
outside that directory is rejected with an error JSON rather than raising an
exception, so the LLM receives a structured refusal it can act on.

Design decisions
----------------
- PROJECT_ROOT is computed once at import time from __file__ so it is
  deterministic regardless of the working directory at runtime.
- No LangGraph or project-internal imports — only stdlib + langchain_core.tools,
  so this module is importable anywhere.
- Large binary/data files (.parquet, .csv, .wav, etc.) are skipped in
  list_project_files to keep the file list concise and LLM-friendly.
- read_file_section caps reads at 200 lines to prevent context flooding.
"""

import json
import os
from pathlib import Path

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Project root — resolves to NHTSA_Project/ regardless of cwd.
# __file__ is Project_Tools/Codebase_Tools.py, so .parent.parent is NHTSA_Project/.
# ---------------------------------------------------------------------------
PROJECT_ROOT = (Path(__file__).parent.parent).resolve()

# Extensions we skip when listing files — these are large data or audio files
# that would clutter the listing and provide no value to the LLM.
_SKIP_EXTENSIONS = {".parquet", ".csv", ".wav", ".mp3", ".npy", ".pkl"}

# Maximum number of file paths returned by list_project_files before truncation.
_MAX_FILES = 500

# Maximum number of lines that read_file_section will return in one call.
_MAX_LINES = 200


def _is_within_root(target: Path) -> bool:
    """
    Return True if `target` is the project root itself or a descendant of it.

    Uses Path.is_relative_to (Python 3.9+).  Both paths must already be
    resolved (absolute, symlinks collapsed) before calling this helper.
    """
    # is_relative_to returns True when target == PROJECT_ROOT too, which is
    # the correct behaviour for listing the root directory itself.
    return target == PROJECT_ROOT or target.is_relative_to(PROJECT_ROOT)


@tool
def list_project_files(subdir: str = "") -> str:
    """
    List source files inside the NHTSA_Project directory tree.

    Use this tool to navigate the codebase — discover which modules exist,
    understand the directory layout, or find a file before calling
    read_file_section.  The listing is read-only and scoped strictly to
    NHTSA_Project/; paths outside that root are rejected.

    Large data and audio files are excluded from the listing:
      .parquet, .csv, .wav, .mp3, .npy, .pkl

    __pycache__ directories are skipped entirely.

    Parameters
    ----------
    subdir : str, optional
        Subdirectory relative to NHTSA_Project/ to list (e.g. "LLM_Tools" or
        "Project_Tools").  Defaults to "" which lists the entire project tree.

    Returns
    -------
    str
        JSON object with keys:
          - "root"        : str        — absolute path of NHTSA_Project/
          - "subdir"      : str        — the subdir argument as supplied
          - "files"       : list[str]  — file paths relative to PROJECT_ROOT,
                                         sorted alphabetically, capped at 500
          - "_truncated"  : bool       — true if more than 500 files exist
        On error:
          - "error"       : str        — human-readable reason for failure
    """
    # Resolve the target directory, collapsing any ".." segments so that
    # the escape check below operates on canonical absolute paths.
    target = (PROJECT_ROOT / subdir).resolve()

    # Security gate: reject any path that resolves outside the project root.
    # This prevents prompt-injection attacks that try to read /etc/passwd etc.
    if not _is_within_root(target):
        return json.dumps({"error": "path escapes project root"})

    collected: list[str] = []
    truncated = False

    for dirpath, dirnames, filenames in os.walk(target):
        # Prune __pycache__ in-place so os.walk does not descend into them.
        # Modifying dirnames[:] (not dirnames =) is the documented os.walk pattern.
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]

        for fname in filenames:
            # Skip data/audio/binary files that are useless to the LLM.
            ext = Path(fname).suffix.lower()
            if ext in _SKIP_EXTENSIONS:
                continue

            full_path = Path(dirpath) / fname
            # Return paths relative to PROJECT_ROOT so they are portable and
            # can be passed directly to read_file_section.
            rel = str(full_path.relative_to(PROJECT_ROOT))
            collected.append(rel)

            if len(collected) >= _MAX_FILES:
                truncated = True
                break  # stop collecting; we'll break the outer loop too

        if truncated:
            break

    collected.sort()

    return json.dumps({
        "root": str(PROJECT_ROOT),
        "subdir": subdir,
        "files": collected,
        "_truncated": truncated,
    })


@tool
def read_file_section(path: str, start_line: int, end_line: int) -> str:
    """
    Read a specific line range from a source file inside NHTSA_Project/.

    Use this tool to inspect a module, function, or config block after finding
    the file with list_project_files.  All paths must be relative to
    NHTSA_Project/ (as returned by list_project_files).  The tool is read-only
    and scoped strictly to NHTSA_Project/; paths outside are rejected.

    Lines are 1-indexed (line 1 is the first line of the file).  A maximum of
    200 lines can be returned per call; request a narrower range or make
    multiple calls to read a larger region.

    Parameters
    ----------
    path : str
        File path relative to NHTSA_Project/ (e.g. "LLM_Tools/NHTSA_Query_Tools.py").
    start_line : int
        First line to return (1-indexed, inclusive).  Values below 1 are
        clamped to 1 automatically.
    end_line : int
        Last line to return (1-indexed, inclusive).  Must satisfy
        end_line - start_line <= 200; larger ranges are rejected with an error.

    Returns
    -------
    str
        JSON object with keys:
          - "path"  : str              — the path argument as supplied
          - "lines" : list[object]     — list of {"n": <line_number>, "text": <line>}
                                         objects for lines start_line..end_line
                                         that exist in the file
        On error:
          - "error" : str              — human-readable reason:
              "path escapes project root" | "file not found" | "range exceeds 200-line cap"
    """
    # Resolve to an absolute canonical path before the escape check.
    target = (PROJECT_ROOT / path).resolve()

    # Reject paths that escape the project root.
    if not _is_within_root(target):
        return json.dumps({"error": "path escapes project root"})

    # Reject missing files with a structured error so the LLM can correct the path.
    if not target.is_file():
        return json.dumps({"error": "file not found", "path": path})

    # Clamp start_line to a minimum of 1 (files are 1-indexed).
    if start_line < 1:
        start_line = 1

    # Enforce the 200-line cap before reading anything — avoids loading a
    # large file only to reject the request after the fact.
    if end_line - start_line > _MAX_LINES:
        return json.dumps({"error": "range exceeds 200-line cap"})

    # Read the file with UTF-8 encoding; replace undecodable bytes rather than
    # raising, so binary-ish files (e.g. a .py with a stray latin-1 char) are
    # still partially readable instead of crashing the tool call.
    text = target.read_text(encoding="utf-8", errors="replace")
    file_lines = text.splitlines()

    # Build the result as a list of {"n": line_number, "text": line_content}
    # objects.  Line numbers are 1-based to match editor conventions and the
    # parameter semantics documented above.
    lines = [
        {"n": i, "text": line}
        for i, line in enumerate(file_lines, start=1)
        if start_line <= i <= end_line
    ]

    return json.dumps({"path": path, "lines": lines})
