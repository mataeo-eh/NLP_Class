"""
Chart_Tools.py
--------------
SANCTIONED non-Jupyter consumer of the chart functions in Create_Charts.py.

WHY THIS MODULE EXISTS
----------------------
The NHTSA project's base rule (documented in CLAUDE.md, "Chart Functions" section)
requires that chart functions render via plt.show() inside Charts.ipynb — they must
never save to disk and must not be consumed outside Jupyter.

This module is the ONE exception to that rule, as formally documented in
CLAUDE.md under "LangGraph Pipeline Chart Exception".  It exists solely to give
the LangGraph agentic_explore node the ability to trigger chart rendering and
receive structured chart data that the model can reason about — without a Jupyter
kernel in the loop.

DESIGN CONSTRAINTS
------------------
- Sets matplotlib's backend to "Agg" (non-interactive, off-screen renderer) BEFORE
  any pyplot import.  Agg is process-global; once set it cannot be changed without
  restarting the interpreter.  All chart creation in this process will use Agg —
  which means Charts.ipynb must NOT be imported into the same process as this module.
  The LangGraph pipeline runs in its own process, so this is safe.

- Charts are rendered to in-memory PNG bytes via BytesIO (no disk write on macOS).

- Display path:
    macOS  → subprocess.Popen(["open", "-a", "Preview", "-f"], stdin=PIPE)
             PNG bytes are piped to Preview's stdin.  No temp file is created.
    Linux  → write to a NamedTemporaryFile, then xdg-open.
    Windows→ write to a NamedTemporaryFile, then os.startfile.

- Temp files created on non-macOS platforms are tracked in _TEMP_PATHS and cleaned
  up by cleanup_temp_charts() which the agentic node calls at end-of-loop.

- Each @tool wrapper stores a structured data summary in _LAST_CHART_DATA so the
  LLM can "see" what the chart contains even though it cannot see the rendered PNG.
  The model should call describe_chart_data() to retrieve this summary.

See also: Unit2/NHTSA_Project/CLAUDE.md — "LangGraph Pipeline Chart Exception".
"""

import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from LLM_Tools.MCP_To_Tools import register_stats_dataset

# is_headless_mode is consulted at the top of every chart @tool so the FastAPI
# backend never touches matplotlib at all — the server has no display and no
# memory budget for a 150-MB import that would only render a PNG nobody can see.
from Project_Tools.Runtime_Options import is_headless_mode

# ---------------------------------------------------------------------------
# Lazy / optional matplotlib + Create_Charts import.
#
# The local CLI ABSOLUTELY needs the original behaviour: matplotlib's "Agg"
# backend MUST be selected BEFORE pyplot is loaded, then Create_Charts is
# imported so its top-level `import matplotlib.pyplot as plt` inherits that
# backend.  See `LangGraph Pipeline Chart Exception` in CLAUDE.md.
#
# The hosted backend on Render's 512 MB free tier cannot afford the import
# (matplotlib alone is ~150 MB resident, and Create_Charts also pulls pandas
# + numpy at module top).  We solve both cases by wrapping the import block
# in try/except:
#
#   - If matplotlib + Create_Charts are present (local venv), the original
#     Agg-before-pyplot ordering is preserved exactly.  Chart tools work.
#
#   - If either import fails (Render, no matplotlib), `_CHART_BACKEND_READY`
#     stays False and every chart @tool returns a structured refusal string
#     instead of trying to render.  This module remains importable so the
#     compiled LangGraph that contains these tools loads cleanly.
#
# We also short-circuit on is_headless_mode() at the top of every @tool, so
# even if a future Render image happens to ship matplotlib, the chart tools
# never actually render server-side.  Belt-and-suspenders.
# ---------------------------------------------------------------------------
plt = None  # type: ignore[assignment]  rebound below if matplotlib is available
create_bar_chart = None  # type: ignore[assignment]
create_human_subsystem_frequency_chart = None  # type: ignore[assignment]
create_model_year_chart = None  # type: ignore[assignment]
compare_LLM_to_NHTSA = None  # type: ignore[assignment]
create_subsystem_frequency_by_make_chart = None  # type: ignore[assignment]
create_subsystem_safety_signal_chart = None  # type: ignore[assignment]
create_model_year_trend_chart = None  # type: ignore[assignment]
_CHART_BACKEND_READY = False
_CHART_BACKEND_ERROR: str | None = None

try:
    import matplotlib  # type: ignore[import-not-found]
    matplotlib.use("Agg")          # must precede "import matplotlib.pyplot as plt"
    import matplotlib.pyplot as plt  # type: ignore[no-redef]  # noqa: E402
    # Make Create_Charts.py importable: insert this file's directory at the
    # front of sys.path so "import Create_Charts" resolves without needing
    # the caller to manage sys.path. Index 0 wins over any stale package
    # entries that might shadow local modules.
    sys.path.insert(0, str(Path(__file__).parent))
    from Create_Charts import (  # type: ignore[no-redef]  # noqa: E402
        create_bar_chart,
        create_human_subsystem_frequency_chart,
        create_model_year_chart,
        compare_LLM_to_NHTSA,
        create_subsystem_frequency_by_make_chart,
        create_subsystem_safety_signal_chart,
        create_model_year_trend_chart,
    )
    _CHART_BACKEND_READY = True
except ImportError as exc:
    _CHART_BACKEND_ERROR = str(exc)


def _headless_chart_refusal(chart_name: str) -> str:
    """
    Return a structured JSON-string refusal that the LLM can read and adapt to,
    instead of trying to render a chart in an environment that cannot display
    one. Mirrors the shape of a successful render's return value (a JSON string)
    so the agentic loop's tool-result parser does not have to special-case this.
    """
    payload = {
        "rendered": False,
        "chart_name": chart_name,
        "reason": (
            "Chart rendering is disabled in this hosted/headless environment. "
            "There is no display to show a PNG, and matplotlib is not installed "
            "on the server. Describe the data in prose instead, or call a data "
            "tool (filter_rows, filter_csv, etc.) to fetch the underlying numbers."
        ),
    }
    return json.dumps(payload)

# ---------------------------------------------------------------------------
# Project-level path constants (resolved at import time so tools don't need
# to accept raw file-system paths from the LLM — that would be a security risk
# and would fragment path logic across the codebase).
# ---------------------------------------------------------------------------

# Parquet database built by NHTSA/Build_DF.py
PARQUET_PATH = Path(__file__).parent.parent / "NHTSA" / "complaints_cleaned.parquet"

# Directory where pipeline CSV outputs are written (Specific_Subsystem_Prompt.csv etc.)
OUTPUTS_DIR = Path(__file__).parent.parent / "Outputs"

# ---------------------------------------------------------------------------
# Module-level mutable state
# ---------------------------------------------------------------------------

# Populated by each @tool wrapper when it renders a chart.
# Key   = chart_name string (e.g. "create_model_year_chart").
# Value = dict of structured summary data the LLM can reason about.
# The LLM cannot see PNG bytes; it reads this dict via describe_chart_data().
_LAST_CHART_DATA: dict[str, dict] = {}

# Tracks temp PNG file paths created on non-macOS platforms so cleanup_temp_charts()
# can delete them at end-of-loop.  macOS uses Preview stdin piping → no temp files.
_TEMP_PATHS: list[str] = []


# ---------------------------------------------------------------------------
# Private rendering helpers (not @tool-decorated — internal use only)
# ---------------------------------------------------------------------------

def _render_fig_to_png(fig) -> bytes:
    """
    Render a matplotlib Figure to PNG bytes using an in-memory buffer.

    Why BytesIO:
      The Agg backend writes pixel data to a buffer rather than a file.
      BytesIO gives us a file-like object that fig.savefig() can write to,
      without touching the file system.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to render.  Must have been created in the current process
        with the Agg backend active.

    Returns
    -------
    bytes
        Raw PNG bytes ready to be piped to a display process or written to a file.
    """
    buf = BytesIO()
    # bbox_inches="tight" trims whitespace so the chart fills the image.
    # dpi=120 gives a crisp display on Retina/HiDPI screens without being huge.
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)   # release matplotlib's reference to the figure to free memory
    buf.seek(0)
    return buf.read()


def _display_png_bytes(png: bytes, chart_name: str) -> str:
    """
    Display a PNG (given as raw bytes) using the platform's native viewer.

    Platform behaviour
    ------------------
    macOS   : Pipe PNG bytes to `open -a Preview -f` via stdin.
              Preview opens immediately; we do NOT wait for it to close.
              No temp file is written to disk.
    Linux   : Write to a NamedTemporaryFile (suffix=".png", delete=False),
              append path to _TEMP_PATHS, launch xdg-open.
    Windows : Same temp-file flow as Linux, then os.startfile().

    Parameters
    ----------
    png : bytes
        Raw PNG bytes, typically from _render_fig_to_png().
    chart_name : str
        Human-readable name used in the return message and (on non-macOS) in
        the temp-file name for easier identification.

    Returns
    -------
    str
        Short status string describing how the chart was displayed.
        Stored in the tool's JSON return under the "display" key.
    """
    system = platform.system()

    if system == "Darwin":
        # macOS: pipe PNG bytes directly into Preview's stdin.
        # "open -a Preview -f" tells macOS to open the file from stdin using Preview.
        # We do NOT call proc.wait() — Preview is a GUI app that stays open until the
        # user closes it; waiting would block the LangGraph loop indefinitely.
        proc = subprocess.Popen(
            ["open", "-a", "Preview", "-f"],
            stdin=subprocess.PIPE,
        )
        proc.stdin.write(png)
        proc.stdin.close()
        # proc.stdin is closed; Preview has received the full PNG.
        # Do not call proc.wait() — see note above.
        return "displayed via Preview"

    elif system == "Linux":
        # Linux: write to a temp file, then launch xdg-open.
        # delete=False because xdg-open needs the file to persist until the viewer
        # opens it.  cleanup_temp_charts() will delete it at end-of-loop.
        with tempfile.NamedTemporaryFile(
            suffix=".png", prefix=f"nhtsa_{chart_name}_", delete=False
        ) as tmp:
            tmp.write(png)
            path = tmp.name
        _TEMP_PATHS.append(path)
        subprocess.Popen(["xdg-open", path])
        return f"displayed at {path}"

    else:
        # Windows (or any other platform): temp file + os.startfile.
        with tempfile.NamedTemporaryFile(
            suffix=".png", prefix=f"nhtsa_{chart_name}_", delete=False
        ) as tmp:
            tmp.write(png)
            path = tmp.name
        _TEMP_PATHS.append(path)
        os.startfile(path)  # type: ignore[attr-defined]  # only exists on Windows
        return f"displayed at {path}"


# ---------------------------------------------------------------------------
# Public cleanup helper (called by the agentic node at end-of-loop)
# ---------------------------------------------------------------------------

def cleanup_temp_charts() -> None:
    """
    Delete any temp PNG files that were written to disk on non-macOS platforms.

    When to call:
      The agentic_explore node should call this at the very end of each loop
      iteration, after the model has had a chance to read the chart summary.

    macOS note:
      On macOS, _display_png_bytes() pipes bytes directly to Preview without
      creating a file, so _TEMP_PATHS will be empty and this function is a no-op.

    Side effects:
      - Deletes files listed in _TEMP_PATHS that still exist on disk.
      - Clears _TEMP_PATHS so the list doesn't grow across loop iterations.
    """
    for path in _TEMP_PATHS:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass   # best-effort cleanup; don't crash the loop over a temp file
    _TEMP_PATHS.clear()


# ---------------------------------------------------------------------------
# @tool-decorated wrappers
# ---------------------------------------------------------------------------

@tool
def create_bar_chart_tool(columns: list[str]) -> str:
    """
    Render a bar chart showing complaint counts per unique value for the given
    CSV column(s) from the Specific_Subsystem_Prompt.csv output file.

    The underlying create_bar_chart() function saves one PNG per column to
    Outputs/charts/ AND returns a Figure object.  This tool captures the last
    Figure rendered (i.e. the chart for the final column in the list), renders it
    to an in-memory PNG with the Agg backend, and displays it via macOS Preview
    (or an OS-native opener on other platforms).

    TIP: Pass a single column per call so the returned summary and displayed chart
    correspond 1-to-1.  Example: columns=["Specific_Subsystem_Prompt"].

    IMPORTANT — you CANNOT see the rendered chart image.
    After calling this tool, read the structured summary in the returned JSON
    (key: "summary") or call describe_chart_data("create_bar_chart") to retrieve
    it again.  Discuss the chart with the user by describing the summary data.

    Parameters
    ----------
    columns : list of str
        Column name(s) in the CSV to plot.  Each column produces a bar chart of
        value_counts().  The tool returns data for the last column rendered.

    Returns
    -------
    str
        JSON string with keys:
          "chart_name"  : "create_bar_chart"
          "display"     : how the chart was opened ("displayed via Preview" on macOS)
          "summary"     : {
              "csv_path"       : absolute path to the CSV that was read,
              "columns_plotted": list of column names that were found and plotted,
              "value_counts"   : {col_name: {value: count, ...}, ...}
                                 top-10 values per column by frequency
          }
    """
    chart_name = "create_bar_chart"
    csv_path = OUTPUTS_DIR / "Specific_Subsystem_Prompt.csv"

    # --- summary (always computed — drives frontend chart rendering) ---------
    import pandas as pd
    summary: dict = {
        "csv_path": str(csv_path),
        "columns_plotted": [],
        "value_counts": {},
    }
    try:
        df = pd.read_csv(csv_path)
        for col in columns:
            if col in df.columns:
                summary["columns_plotted"].append(col)
                vc = df[col].value_counts().head(10)
                summary["value_counts"][col] = {str(k): int(v) for k, v in vc.items()}
    except Exception as e:
        summary["error"] = str(e)

    _LAST_CHART_DATA[chart_name] = summary

    # Headless / no matplotlib: return summary without rendering.
    # The web frontend will render the chart from the structured data above.
    if is_headless_mode() or not _CHART_BACKEND_READY:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    # --- local CLI: render and display -------------------------------------
    fig = create_bar_chart(str(csv_path), columns, str(OUTPUTS_DIR))

    if fig is None:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    png = _render_fig_to_png(fig)
    display_result = _display_png_bytes(png, chart_name)

    return json.dumps({
        "chart_name": chart_name,
        "rendered": True,
        "display": display_result,
        "summary": _LAST_CHART_DATA[chart_name],
    })


@tool
def create_model_year_chart_tool(include_year: bool = False) -> str:
    """
    Render a horizontal bar chart of the top 30 vehicles (Make · Model, or
    Make · Model · Year) by total NHTSA complaint count.

    Data source: complaints_cleaned.parquet (all complaints, not just sampled ones).
    The chart is designed to reveal which vehicle lines generate the most complaints.

    IMPORTANT — you CANNOT see the rendered chart image.
    After calling this tool, read the structured summary in the returned JSON
    (key: "summary") or call describe_chart_data("create_model_year_chart") to
    retrieve it again.  Use the summary to describe the chart to the user.

    Parameters
    ----------
    include_year : bool, default False
        When False (default): groups by Make + Model only (broader view).
        When True:            groups by Make + Model + Year (finer granularity).

    Returns
    -------
    str
        JSON string with keys:
          "chart_name"  : "create_model_year_chart"
          "display"     : how the chart was opened ("displayed via Preview" on macOS)
          "summary"     : {
              "include_year"      : bool (matches the parameter),
              "top_5_make_model"  : [{"vehicle": str, "count": int}, ...],
              "total_rows"        : int (total complaints in the parquet)
          }
    """
    chart_name = "create_model_year_chart"

    # --- summary (always computed — drives frontend chart rendering) ---------
    import pandas as pd
    import numpy as np

    summary: dict = {
        "include_year": include_year,
        "top_5_make_model": [],
        "total_rows": 0,
    }
    try:
        df = pd.read_parquet(PARQUET_PATH)
        summary["total_rows"] = len(df)

        make_col  = df.columns[3]
        model_col = df.columns[4]
        year_col  = df.columns[5]

        make_model = (
            df[make_col].astype(str).str.strip().str.upper()
            + " "
            + df[model_col].astype(str).str.strip().str.upper()
        )

        if include_year:
            def _extract_year_safe(val) -> str:
                if isinstance(val, np.ndarray):
                    if val.size == 0:
                        return "UNKNOWN"
                    val = val.flat[0]
                elif isinstance(val, list):
                    if not val:
                        return "UNKNOWN"
                    val = val[0]
                if pd.isna(val):
                    return "UNKNOWN"
                year = int(val)
                return "UNKNOWN" if year == 9999 else str(year)

            year_str = df[year_col].apply(_extract_year_safe)
            vehicle_key = make_model + " " + year_str
        else:
            vehicle_key = make_model

        top5 = vehicle_key.value_counts().head(5)
        summary["top_5_make_model"] = [
            {"vehicle": str(v), "count": int(c)} for v, c in top5.items()
        ]
    except Exception as e:
        summary["error"] = str(e)

    _LAST_CHART_DATA[chart_name] = summary

    # Headless / no matplotlib: return summary without rendering.
    if is_headless_mode() or not _CHART_BACKEND_READY:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    # --- local CLI: render and display -------------------------------------
    fig = create_model_year_chart(str(PARQUET_PATH), include_year=include_year)

    if fig is None:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    png = _render_fig_to_png(fig)
    display_result = _display_png_bytes(png, chart_name)

    return json.dumps({
        "chart_name": chart_name,
        "rendered": True,
        "display": display_result,
        "summary": _LAST_CHART_DATA[chart_name],
    })


@tool
def create_human_subsystem_frequency_chart_tool(
    filters: dict | None = None,
    top_n: int = 15,
) -> str:
    """
    Render a bar chart of the most frequent human-labelled subsystem components.

    Data source: complaints_cleaned.parquet. The human labels come from COMPDESC,
    the NHTSA component category field. If the user's query names a make, model,
    crash/fire flag, state, year, or other database field, translate that request
    into the filters dict before calling this tool. If the user asks for the
    overall database distribution, pass filters={} or omit filters.

    Examples:
      filters={"MAKETXT": "TOYOTA"}
      filters={"MAKETXT": "FORD", "CRASH": "Y"}
      filters={"COMPDESC": "SERVICE BRAKES"}

    IMPORTANT: you CANNOT see the rendered chart image. Use the returned
    summary to describe the chart to the user.

    Parameters
    ----------
    filters : dict | None
        Column-to-value equality filters. List-like columns match by membership.
    top_n : int, default 15
        Number of top component labels to plot. Capped internally at 30.

    Returns
    -------
    str
        JSON string with keys:
          "chart_name" : "create_human_subsystem_frequency_chart"
          "display"    : how the chart was opened
          "summary"    : {
              "filters"          : dict,
              "top_n"            : int,
              "matching_rows"    : int,
              "component_counts" : [{"component": str, "count": int}, ...]
          }
    """
    chart_name = "create_human_subsystem_frequency_chart"
    filters = filters or {}
    top_n = max(1, min(int(top_n), 30))

    import pandas as pd
    import numpy as np

    def _is_missing_cell(value):
        if value is None:
            return True
        if isinstance(value, np.ndarray):
            return value.size == 0
        if isinstance(value, list):
            return len(value) == 0
        try:
            return pd.isna(value)
        except (TypeError, ValueError):
            return False

    def _cell_matches_filter(value, expected):
        expected_values = expected if isinstance(expected, list) else [expected]
        expected_strings = {str(v).strip().upper() for v in expected_values}
        if isinstance(value, np.ndarray):
            actual_values = value.tolist()
        elif isinstance(value, list):
            actual_values = value
        else:
            actual_values = [value]
        actual_strings = {str(v).strip().upper() for v in actual_values}
        return bool(actual_strings & expected_strings)

    summary: dict = {
        "filters": filters,
        "top_n": top_n,
        "matching_rows": 0,
        "component_counts": [],
    }

    try:
        df = pd.read_parquet(PARQUET_PATH)
        filtered = df
        for col, expected in filters.items():
            if col not in filtered.columns:
                summary.setdefault("warnings", []).append(
                    f"Column '{col}' not found; filter skipped."
                )
                continue
            filtered = filtered[filtered[col].apply(lambda value: _cell_matches_filter(value, expected))]
            if filtered.empty:
                break

        summary["matching_rows"] = int(len(filtered))

        labels = []
        if "COMPDESC" in filtered.columns:
            for value in filtered["COMPDESC"]:
                if _is_missing_cell(value):
                    continue
                if isinstance(value, np.ndarray):
                    values = value.tolist()
                elif isinstance(value, list):
                    values = value
                else:
                    values = [value]
                for label in values:
                    if _is_missing_cell(label):
                        continue
                    cleaned = str(label).strip().upper()
                    if cleaned:
                        labels.append(cleaned)

        counts = pd.Series(labels).value_counts().head(top_n)
        summary["component_counts"] = [
            {"component": str(component), "count": int(count)}
            for component, count in counts.items()
        ]
    except Exception as e:
        summary["error"] = str(e)

    _LAST_CHART_DATA[chart_name] = summary

    # Headless / no matplotlib: return summary without rendering.
    if is_headless_mode() or not _CHART_BACKEND_READY:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    # --- local CLI: render and display -------------------------------------
    fig = create_human_subsystem_frequency_chart(
        str(PARQUET_PATH),
        filters=filters,
        top_n=top_n,
    )

    if fig is None:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    png = _render_fig_to_png(fig)
    display_result = _display_png_bytes(png, chart_name)

    return json.dumps({
        "chart_name": chart_name,
        "rendered": True,
        "display": display_result,
        "summary": _LAST_CHART_DATA[chart_name],
    })


def _build_csv_parquet_label_comparison(
    *,
    filename: str,
    llm_label_column: str,
    parquet_label_column: str,
    index_column: str,
    complaint_text_column: str,
    top_n: int,
    example_limit: int,
) -> dict:
    """Compute an exact CSV-to-parquet label comparison summary.

    This helper is shared by the data-first comparison tool and the existing
    chart tool so both surfaces use identical join logic, label normalization,
    and agreement definitions.
    """
    import ast
    import pandas as pd
    import numpy as np

    csv_path = OUTPUTS_DIR / filename
    summary: dict = {
        "filename": filename,
        "join": {
            "index_column": index_column,
            "llm_label_column": llm_label_column,
            "parquet_label_column": parquet_label_column,
        },
        "compared_rows": 0,
        "skipped_rows": {
            "missing_index": 0,
            "missing_parquet_row": 0,
            "missing_csv_label_column": 0,
            "missing_parquet_label_column": 0,
        },
        "agreement_counts": {"full": 0, "partial": 0, "none": 0},
        "percentages": {"full": 0.0, "partial": 0.0, "none": 0.0},
        "label_metrics": {
            "true_positive_labels": 0,
            "false_positive_labels": 0,
            "false_negative_labels": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        },
        "top_human_label_accuracy": [],
        "top_confused_pairs": [],
        "examples": {"full": [], "partial": [], "none": []},
    }

    if not csv_path.exists():
        summary["error"] = f"CSV file '{filename}' was not found in the Outputs directory."
        return summary

    try:
        csv_df = pd.read_csv(csv_path)
        pq_df = pd.read_parquet(PARQUET_PATH)
    except Exception as exc:
        summary["error"] = str(exc)
        return summary

    required_csv_columns = [index_column, llm_label_column]
    missing_csv_columns = [col for col in required_csv_columns if col not in csv_df.columns]
    if missing_csv_columns:
        summary["error"] = (
            "CSV file is missing one or more required columns for comparison."
        )
        summary["missing_csv_columns"] = missing_csv_columns
        return summary

    if parquet_label_column not in pq_df.columns:
        summary["error"] = (
            f"Parquet column '{parquet_label_column}' was not found in complaints_cleaned.parquet."
        )
        return summary

    top_n = max(1, min(int(top_n), 25))
    example_limit = max(0, min(int(example_limit), 10))

    def _coerce_index(raw_value) -> int | None:
        if raw_value is None:
            return None
        text_value = str(raw_value).strip()
        if not text_value:
            return None
        try:
            return int(text_value)
        except ValueError:
            try:
                return int(float(text_value))
            except ValueError:
                return None

    def _normalize_labels(raw_value, *, parse_string_literal: bool) -> frozenset[str]:
        if raw_value is None:
            return frozenset()
        if isinstance(raw_value, np.ndarray):
            values = raw_value.tolist()
        elif isinstance(raw_value, (list, tuple, set)):
            values = list(raw_value)
        elif isinstance(raw_value, str):
            text_value = raw_value.strip()
            if not text_value:
                return frozenset()
            if parse_string_literal:
                try:
                    parsed = ast.literal_eval(text_value)
                except (ValueError, SyntaxError):
                    parsed = [text_value]
            else:
                parsed = [text_value]

            if isinstance(parsed, np.ndarray):
                values = parsed.tolist()
            elif isinstance(parsed, (list, tuple, set)):
                values = list(parsed)
            elif parsed is None:
                values = []
            else:
                values = [parsed]
        else:
            values = [raw_value]

        cleaned = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, float) and pd.isna(value):
                continue
            text_value = str(value).strip().upper()
            if text_value:
                cleaned.append(text_value)
        return frozenset(cleaned)

    def _llm_label_matches_nhtsa_set(llm_label: str, nhtsa_set: frozenset[str]) -> bool:
        for nhtsa_label in nhtsa_set:
            if llm_label == nhtsa_label:
                return True
            if nhtsa_label.startswith(llm_label + ":") or nhtsa_label.startswith(llm_label + "/"):
                return True
        return False

    def _nhtsa_label_matched_by_llm(nhtsa_label: str, llm_set: frozenset[str]) -> bool:
        for llm_label in llm_set:
            if nhtsa_label == llm_label:
                return True
            if nhtsa_label.startswith(llm_label + ":") or nhtsa_label.startswith(llm_label + "/"):
                return True
        return False

    human_label_stats: dict[str, dict[str, int]] = {}
    miss_pairs: Counter[tuple[str, str]] = Counter()
    tp_labels = fp_labels = fn_labels = 0

    for _, row in csv_df.iterrows():
        df_index = _coerce_index(row.get(index_column))
        if df_index is None:
            summary["skipped_rows"]["missing_index"] += 1
            continue
        if df_index not in pq_df.index:
            summary["skipped_rows"]["missing_parquet_row"] += 1
            continue

        llm_set = _normalize_labels(row.get(llm_label_column), parse_string_literal=True)
        nhtsa_set = _normalize_labels(
            pq_df.at[df_index, parquet_label_column],
            parse_string_literal=False,
        )

        if not llm_set:
            summary["skipped_rows"]["missing_csv_label_column"] += 1
            continue
        if not nhtsa_set:
            summary["skipped_rows"]["missing_parquet_label_column"] += 1
            continue

        matched_human = {
            label for label in nhtsa_set if _nhtsa_label_matched_by_llm(label, llm_set)
        }
        matched_llm = {
            label for label in llm_set if _llm_label_matches_nhtsa_set(label, nhtsa_set)
        }

        tp_labels += len(matched_human)
        fn_labels += len(nhtsa_set - matched_human)
        fp_labels += len(llm_set - matched_llm)

        for nhtsa_label in nhtsa_set:
            stats = human_label_stats.setdefault(nhtsa_label, {"total": 0, "matched": 0})
            stats["total"] += 1
            if nhtsa_label in matched_human:
                stats["matched"] += 1

        if llm_set == nhtsa_set:
            outcome = "full"
        elif matched_human:
            outcome = "partial"
        else:
            outcome = "none"
            for nhtsa_label in nhtsa_set:
                for llm_label in llm_set:
                    miss_pairs[(nhtsa_label, llm_label)] += 1

        summary["agreement_counts"][outcome] += 1
        summary["compared_rows"] += 1

        if len(summary["examples"][outcome]) < example_limit:
            complaint_text = ""
            if complaint_text_column in pq_df.columns:
                raw_text = pq_df.at[df_index, complaint_text_column]
                complaint_text = "" if raw_text is None else str(raw_text)
            summary["examples"][outcome].append(
                {
                    "df_index": int(df_index),
                    "human_labels": sorted(nhtsa_set),
                    "llm_labels": sorted(llm_set),
                    "complaint_text": complaint_text,
                }
            )

    compared = summary["compared_rows"]
    if compared > 0:
        for key, count in summary["agreement_counts"].items():
            summary["percentages"][key] = round(count / compared * 100, 1)

    precision = tp_labels / (tp_labels + fp_labels) if (tp_labels + fp_labels) else 0.0
    recall = tp_labels / (tp_labels + fn_labels) if (tp_labels + fn_labels) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    summary["label_metrics"] = {
        "true_positive_labels": tp_labels,
        "false_positive_labels": fp_labels,
        "false_negative_labels": fn_labels,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

    label_accuracy = []
    for label, stats in human_label_stats.items():
        total = stats["total"]
        matched = stats["matched"]
        label_accuracy.append(
            {
                "label": label,
                "total": total,
                "matched": matched,
                "missed": total - matched,
                "accuracy_pct": round(matched / total * 100, 1) if total else 0.0,
            }
        )
    label_accuracy.sort(key=lambda item: (-item["total"], item["label"]))
    summary["top_human_label_accuracy"] = label_accuracy[:top_n]
    summary["top_confused_pairs"] = [
        {"nhtsa_label": nhtsa_label, "llm_label": llm_label, "count": count}
        for (nhtsa_label, llm_label), count in miss_pairs.most_common(top_n)
    ]

    return summary


def _to_stats_cell(value):
    """Convert one comparison cell into a scalar JSON value for RMCP datasets."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), default=str)
    try:
        import numpy as np
        import pandas as pd

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and pd.isna(value):
            return None
    except Exception:
        pass
    return value


@tool
def compare_csv_to_parquet_labels(
    filename: str = "Specific_Subsystem_Prompt.csv",
    llm_label_column: str = "subsystems",
    parquet_label_column: str = "COMPDESC",
    index_column: str = "df_index",
    complaint_text_column: str = "CDESCR",
    top_n: int = 10,
    example_limit: int = 5,
) -> str:
    """Compare CSV labels against human parquet labels via the stored row index.

    Use this for exact parquet-to-CSV evaluation. It scans every row in the CSV,
    joins back to the parquet on df_index, and returns exact agreement counts,
    precision/recall/F1, per-label accuracy, confusion pairs, and example rows.
    """
    summary = _build_csv_parquet_label_comparison(
        filename=filename,
        llm_label_column=llm_label_column,
        parquet_label_column=parquet_label_column,
        index_column=index_column,
        complaint_text_column=complaint_text_column,
        top_n=top_n,
        example_limit=example_limit,
    )
    return json.dumps(summary, indent=2)


@tool
def build_csv_parquet_label_stats_dataset(
    filename: str = "Specific_Subsystem_Prompt.csv",
    llm_label_column: str = "subsystems",
    parquet_label_column: str = "COMPDESC",
    index_column: str = "df_index",
    include_parquet_columns: list[str] | None = None,
) -> str:
    """
    Store an exact CSV-to-parquet comparison table server-side for RMCP stats tools.

    The returned dataset contains one row per compared complaint plus derived
    agreement features. Use the returned `dataset_id` with wrapped RMCP tools
    such as chi_square_test, t_test, anova, or correlation_analysis.
    """
    import ast
    import pandas as pd

    csv_path = OUTPUTS_DIR / filename
    if not csv_path.exists():
        return json.dumps(
            {"error": f"CSV file '{filename}' was not found in the Outputs directory."},
            indent=2,
        )

    csv_df = pd.read_csv(csv_path)
    pq_df = pd.read_parquet(PARQUET_PATH)
    include_parquet_columns = [
        column for column in dict.fromkeys(include_parquet_columns or []) if column
    ]

    required_csv_columns = [index_column, llm_label_column]
    missing_csv_columns = [col for col in required_csv_columns if col not in csv_df.columns]
    if missing_csv_columns:
        return json.dumps(
            {
                "error": "CSV file is missing one or more required columns for comparison.",
                "missing_csv_columns": missing_csv_columns,
            },
            indent=2,
        )

    missing_parquet_columns = [
        col
        for col in [parquet_label_column, *include_parquet_columns]
        if col not in pq_df.columns
    ]
    if missing_parquet_columns:
        return json.dumps(
            {
                "error": "One or more requested parquet columns were not found.",
                "missing_parquet_columns": missing_parquet_columns,
            },
            indent=2,
        )

    def _coerce_index(raw_value) -> int | None:
        if raw_value is None:
            return None
        text_value = str(raw_value).strip()
        if not text_value:
            return None
        try:
            return int(text_value)
        except ValueError:
            try:
                return int(float(text_value))
            except ValueError:
                return None

    def _normalize_labels(raw_value, *, parse_string_literal: bool) -> frozenset[str]:
        if raw_value is None:
            return frozenset()
        if isinstance(raw_value, str):
            text_value = raw_value.strip()
            if not text_value:
                return frozenset()
            if parse_string_literal:
                try:
                    parsed = ast.literal_eval(text_value)
                except (ValueError, SyntaxError):
                    parsed = [text_value]
            else:
                parsed = [text_value]
        elif isinstance(raw_value, (list, tuple, set)):
            parsed = list(raw_value)
        else:
            parsed = [raw_value]

        cleaned = []
        for value in parsed:
            if value is None:
                continue
            text_value = str(value).strip().upper()
            if text_value and text_value != "NAN":
                cleaned.append(text_value)
        return frozenset(cleaned)

    def _llm_matches_human(llm_label: str, human_set: frozenset[str]) -> bool:
        for human_label in human_set:
            if llm_label == human_label:
                return True
            if human_label.startswith(llm_label + ":") or human_label.startswith(llm_label + "/"):
                return True
        return False

    def _human_matched_by_llm(human_label: str, llm_set: frozenset[str]) -> bool:
        for llm_label in llm_set:
            if human_label == llm_label:
                return True
            if human_label.startswith(llm_label + ":") or human_label.startswith(llm_label + "/"):
                return True
        return False

    rows: list[dict[str, Any]] = []
    skipped = {
        "missing_index": 0,
        "missing_parquet_row": 0,
        "missing_csv_label_column": 0,
        "missing_parquet_label_column": 0,
    }

    for _, row in csv_df.iterrows():
        df_index = _coerce_index(row.get(index_column))
        if df_index is None:
            skipped["missing_index"] += 1
            continue
        if df_index not in pq_df.index:
            skipped["missing_parquet_row"] += 1
            continue

        llm_set = _normalize_labels(row.get(llm_label_column), parse_string_literal=True)
        human_set = _normalize_labels(
            pq_df.at[df_index, parquet_label_column],
            parse_string_literal=False,
        )
        if not llm_set:
            skipped["missing_csv_label_column"] += 1
            continue
        if not human_set:
            skipped["missing_parquet_label_column"] += 1
            continue

        matched_human = {
            label for label in human_set if _human_matched_by_llm(label, llm_set)
        }
        matched_llm = {
            label for label in llm_set if _llm_matches_human(label, human_set)
        }
        false_positive = llm_set - matched_llm
        false_negative = human_set - matched_human

        if llm_set == human_set:
            agreement_level = "FULL"
        elif matched_human:
            agreement_level = "PARTIAL"
        else:
            agreement_level = "NONE"

        union_count = len(human_set | llm_set)
        overlap_count = len(matched_human)
        row_payload: dict[str, Any] = {
            "df_index": int(df_index),
            "agreement_level": agreement_level,
            "full_match": int(agreement_level == "FULL"),
            "partial_match": int(agreement_level == "PARTIAL"),
            "no_match": int(agreement_level == "NONE"),
            "human_label_count": len(human_set),
            "llm_label_count": len(llm_set),
            "matched_label_count": overlap_count,
            "false_positive_label_count": len(false_positive),
            "false_negative_label_count": len(false_negative),
            "jaccard_similarity": round(overlap_count / union_count, 6) if union_count else 0.0,
            "human_match_ratio": round(overlap_count / len(human_set), 6) if human_set else 0.0,
            "llm_match_ratio": round(overlap_count / len(llm_set), 6) if llm_set else 0.0,
            "human_primary_label": sorted(human_set)[0],
            "llm_primary_label": sorted(llm_set)[0],
            "human_labels_joined": " | ".join(sorted(human_set)),
            "llm_labels_joined": " | ".join(sorted(llm_set)),
        }
        for column in include_parquet_columns:
            row_payload[column] = _to_stats_cell(pq_df.at[df_index, column])
        rows.append(row_payload)

    if not rows:
        return json.dumps(
            {"error": "No comparison rows were available for stats dataset creation.", "skipped_rows": skipped},
            indent=2,
        )

    columns = list(rows[0].keys())
    data = {column: [row[column] for row in rows] for column in columns}
    manifest = register_stats_dataset(
        data=data,
        source="csv_parquet_label_comparison",
        description="Exact CSV-to-parquet label comparison dataset prepared for hosted statistical tools.",
        metadata={
            "filename": filename,
            "llm_label_column": llm_label_column,
            "parquet_label_column": parquet_label_column,
            "index_column": index_column,
            "include_parquet_columns": include_parquet_columns,
            "skipped_rows": skipped,
        },
    )
    return json.dumps(manifest, indent=2, default=str)


@tool
def compare_LLM_to_NHTSA_tool() -> str:
    """
    Render a three-subplot comparison chart of LLM subsystem predictions vs.
    NHTSA expert (COMPDESC) labels for the sampled complaints in
    Specific_Subsystem_Prompt.csv.

    Subplot 1: Overall agreement breakdown (Full / Partial / No Agreement counts
               and percentages across all sampled complaints).
    Subplot 2: Per-NHTSA-category accuracy (top-20 categories; stacked bar showing
               how many complaint-label pairs the LLM got right vs missed).
    Subplot 3: Miss confusion heatmap (for "No Agreement" complaints; shows which
               LLM labels were substituted for which NHTSA ground-truth labels).

    IMPORTANT — you CANNOT see the rendered chart image.
    After calling this tool, read the structured summary in the returned JSON
    (key: "summary") or call describe_chart_data("compare_LLM_to_NHTSA") to
    retrieve it again.  Use the summary to describe findings to the user.

    Parameters
    ----------
    (none — paths are resolved from project constants)

    Returns
    -------
    str
        JSON string with keys:
          "chart_name"  : "compare_LLM_to_NHTSA"
          "display"     : how the chart was opened ("displayed via Preview" on macOS)
          "summary"     : {
              "total_complaints"  : int,
              "agreement_counts"  : {"full": int, "partial": int, "none": int},
              "percentages"       : {"full": float, "partial": float, "none": float},
              "top_5_confused_pairs": [
                  {"nhtsa_label": str, "llm_label": str, "count": int}, ...
              ]
          }
    """
    chart_name = "compare_LLM_to_NHTSA"

    # --- summary (always computed — drives frontend chart rendering) ---------
    summary = _build_csv_parquet_label_comparison(
        filename="Specific_Subsystem_Prompt.csv",
        llm_label_column="subsystems",
        parquet_label_column="COMPDESC",
        index_column="df_index",
        complaint_text_column="CDESCR",
        top_n=5,
        example_limit=3,
    )

    _LAST_CHART_DATA[chart_name] = summary

    # Headless / no matplotlib: return summary without rendering.
    if is_headless_mode() or not _CHART_BACKEND_READY:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    # --- local CLI: render and display -------------------------------------
    csv_path = OUTPUTS_DIR / "Specific_Subsystem_Prompt.csv"
    fig = compare_LLM_to_NHTSA(str(csv_path), str(PARQUET_PATH))

    if fig is None:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    png = _render_fig_to_png(fig)
    display_result = _display_png_bytes(png, chart_name)

    return json.dumps({
        "chart_name": chart_name,
        "rendered": True,
        "display": display_result,
        "summary": _LAST_CHART_DATA[chart_name],
    })


@tool
def create_subsystem_frequency_by_make_chart_tool(
    makes: list[str] | None = None,
    top_n: int = 8,
) -> str:
    """
    Render a per-make breakdown of the most frequent human-labelled subsystem
    components (COMPDESC) from complaints_cleaned.parquet.

    Each vehicle make gets its own subplot showing the top `top_n` subsystem
    labels for complaints associated with that make. If `makes` is omitted,
    the chart uses the four most common makes in the full database automatically.

    Use this chart when the user wants to compare which subsystem components
    are most reported across different vehicle manufacturers.

    Examples:
      makes=["TOYOTA", "FORD"]           # compare two specific makes
      makes=None                         # auto-select top-4 makes
      makes=["HONDA", "GM", "CHRYSLER", "BMW"]

    IMPORTANT — you CANNOT see the rendered chart image.
    After calling this tool, read the structured summary in the returned JSON
    (key: "summary") or call describe_chart_data("create_subsystem_frequency_by_make_chart")
    to retrieve it again.  Use the summary to describe findings to the user.

    Parameters
    ----------
    makes : list[str] | None
        Vehicle make names to include (case-insensitive). When None, the four
        most common makes in the database are selected automatically.
    top_n : int, default 8
        Number of top subsystem labels to plot per make. Capped internally at 15.

    Returns
    -------
    str
        JSON string with keys:
          "chart_name" : "create_subsystem_frequency_by_make_chart"
          "display"    : how the chart was opened
          "summary"    : {
              "makes_requested" : list[str] | "auto (top 4)",
              "top_n"           : int,
              "makes_plotted"   : [
                  {
                      "make"            : str,
                      "complaint_count" : int,
                      "top_components"  : [{"component": str, "count": int}, ...]
                  }, ...
              ]
          }
    """
    chart_name = "create_subsystem_frequency_by_make_chart"
    top_n = max(1, min(int(top_n), 15))

    import pandas as pd
    import numpy as np

    def _first_clean(value) -> str:
        # Mirror _first_clean_value from Create_Charts for the summary computation.
        if value is None:
            return ""
        if isinstance(value, np.ndarray):
            value = value.flat[0] if value.size > 0 else None
        elif isinstance(value, list):
            value = value[0] if value else None
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip().upper()

    def _flatten_labels(series) -> list[str]:
        labels = []
        for val in series:
            if val is None:
                continue
            if isinstance(val, (np.ndarray, list)):
                items = val.tolist() if isinstance(val, np.ndarray) else val
            else:
                items = [val]
            for item in items:
                try:
                    if pd.isna(item):
                        continue
                except (TypeError, ValueError):
                    pass
                cleaned = str(item).strip().upper()
                if cleaned:
                    labels.append(cleaned)
        return labels

    summary: dict = {
        "makes_requested": makes if makes else "auto (top 4)",
        "top_n": top_n,
        "makes_plotted": [],
    }

    try:
        df = pd.read_parquet(PARQUET_PATH)
        df["_make_key"] = df["MAKETXT"].apply(_first_clean)

        selected_makes = (
            [str(m).strip().upper() for m in makes]
            if makes
            else df["_make_key"].value_counts().head(4).index.tolist()
        )

        for make in selected_makes:
            make_df = df[df["_make_key"] == make]
            labels = _flatten_labels(make_df["COMPDESC"])
            counts = pd.Series(labels).value_counts().head(top_n)
            summary["makes_plotted"].append({
                "make": make,
                "complaint_count": int(len(make_df)),
                "top_components": [
                    {"component": str(comp), "count": int(cnt)}
                    for comp, cnt in counts.items()
                ],
            })
    except Exception as e:
        summary["error"] = str(e)

    _LAST_CHART_DATA[chart_name] = summary

    # Headless / no matplotlib: return summary without rendering.
    if is_headless_mode() or not _CHART_BACKEND_READY:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    # --- local CLI: render and display -------------------------------------
    fig = create_subsystem_frequency_by_make_chart(
        str(PARQUET_PATH),
        makes=makes,
        top_n=top_n,
    )

    if fig is None:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    png = _render_fig_to_png(fig)
    display_result = _display_png_bytes(png, chart_name)

    return json.dumps({
        "chart_name": chart_name,
        "rendered": True,
        "display": display_result,
        "summary": _LAST_CHART_DATA[chart_name],
    })


@tool
def create_subsystem_safety_signal_chart_tool(
    top_n: int = 15,
    min_complaints: int = 100,
) -> str:
    """
    Render a three-panel safety-signal chart breaking down crash rate, fire rate,
    injury rate, and fatality rate by human-labelled subsystem component (COMPDESC).

    Panel 1: Complaint volume (raw count of component-label occurrences).
    Panel 2: Crash rate (%) and fire rate (%) — fraction of complaints flagged
             as involving a crash or fire for that subsystem.
    Panel 3: Injury rate and fatality rate per 1,000 complaints.

    Data source: complaints_cleaned.parquet. Components are ranked by complaint
    volume, then filtered to those with at least `min_complaints` occurrences.
    NHTSA COMPDESC labels are exploded so each complaint may contribute to
    multiple component rows if it carries more than one label.

    Use this chart when the user asks which subsystems are most dangerous,
    most crash-prone, or most associated with injuries and deaths.

    IMPORTANT — you CANNOT see the rendered chart image.
    After calling this tool, read the structured summary in the returned JSON
    (key: "summary") or call describe_chart_data("create_subsystem_safety_signal_chart")
    to retrieve it again.  Use the summary to describe findings to the user.

    Parameters
    ----------
    top_n : int, default 15
        Number of top-volume components to include (after min_complaints filter).
        Capped internally at 25.
    min_complaints : int, default 100
        Minimum number of component-label occurrences a subsystem must have to
        appear in the chart. Filters out low-volume noise.

    Returns
    -------
    str
        JSON string with keys:
          "chart_name" : "create_subsystem_safety_signal_chart"
          "display"    : how the chart was opened
          "summary"    : {
              "top_n"           : int,
              "min_complaints"  : int,
              "components"      : [
                  {
                      "component"            : str,
                      "complaints"           : int,
                      "crash_rate_pct"       : float,
                      "fire_rate_pct"        : float,
                      "injury_rate_per_1k"   : float,
                      "death_rate_per_1k"    : float
                  }, ...
              ]
          }
    """
    chart_name = "create_subsystem_safety_signal_chart"
    top_n = max(1, min(int(top_n), 25))
    min_complaints = max(1, int(min_complaints))

    import pandas as pd
    import numpy as np

    def _is_missing(value) -> bool:
        if value is None:
            return True
        if isinstance(value, (np.ndarray, list)):
            return len(value) == 0
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False

    def _cell_vals(value) -> list:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return value
        return [value]

    def _binary_yes(value) -> bool:
        if isinstance(value, (np.ndarray, list)):
            items = value.tolist() if isinstance(value, np.ndarray) else value
            return any(str(v).strip().upper() == "Y" for v in items if v is not None)
        return str(value).strip().upper() == "Y"

    def _numeric(value) -> float:
        if isinstance(value, (np.ndarray, list)):
            items = value.tolist() if isinstance(value, np.ndarray) else value
            return float(items[0]) if items else 0.0
        try:
            return float(value) if not pd.isna(value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    summary: dict = {
        "top_n": top_n,
        "min_complaints": min_complaints,
        "components": [],
    }

    try:
        df = pd.read_parquet(PARQUET_PATH)
        records = []
        for _, row in df.iterrows():
            for component in _cell_vals(row["COMPDESC"]):
                if _is_missing(component):
                    continue
                cleaned = str(component).strip().upper()
                if not cleaned:
                    continue
                records.append({
                    "component": cleaned,
                    "crash":   int(_binary_yes(row["CRASH"])),
                    "fire":    int(_binary_yes(row["FIRE"])),
                    "injured": _numeric(row["INJURED"]),
                    "deaths":  _numeric(row["DEATHS"]),
                })

        if records:
            long_df = pd.DataFrame(records)
            agg = (
                long_df.groupby("component")
                .agg(
                    complaints=("component", "size"),
                    crashes=("crash", "sum"),
                    fires=("fire", "sum"),
                    injuries=("injured", "sum"),
                    deaths=("deaths", "sum"),
                )
                .reset_index()
            )
            agg = agg[agg["complaints"] >= min_complaints]
            agg = agg.sort_values("complaints", ascending=False).head(top_n)

            for _, r in agg.iterrows():
                n = float(r["complaints"])
                summary["components"].append({
                    "component":          str(r["component"]),
                    "complaints":         int(r["complaints"]),
                    "crash_rate_pct":     round(r["crashes"] / n * 100, 2),
                    "fire_rate_pct":      round(r["fires"] / n * 100, 2),
                    "injury_rate_per_1k": round(r["injuries"] / n * 1000, 2),
                    "death_rate_per_1k":  round(r["deaths"] / n * 1000, 2),
                })
    except Exception as e:
        summary["error"] = str(e)

    _LAST_CHART_DATA[chart_name] = summary

    # Headless / no matplotlib: return summary without rendering.
    if is_headless_mode() or not _CHART_BACKEND_READY:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    # --- local CLI: render and display -------------------------------------
    fig = create_subsystem_safety_signal_chart(
        str(PARQUET_PATH),
        top_n=top_n,
        min_complaints=min_complaints,
    )

    if fig is None:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    png = _render_fig_to_png(fig)
    display_result = _display_png_bytes(png, chart_name)

    return json.dumps({
        "chart_name": chart_name,
        "rendered": True,
        "display": display_result,
        "summary": _LAST_CHART_DATA[chart_name],
    })


@tool
def create_model_year_trend_chart_tool(
    min_year: int = 1990,
    max_year: int | None = None,
) -> str:
    """
    Render a bar chart of NHTSA complaint volume by vehicle model year.

    Each bar represents one model year; height is the number of complaints.
    The NHTSA unknown-year sentinel (9999) is always excluded. The top-5
    highest-volume years are annotated directly on the chart.

    Data source: complaints_cleaned.parquet, YEARTXT column.

    Use this chart when the user wants to know which model years generate the
    most complaints, or to see whether certain manufacturing eras are
    overrepresented in the complaint database.

    IMPORTANT — you CANNOT see the rendered chart image.
    After calling this tool, read the structured summary in the returned JSON
    (key: "summary") or call describe_chart_data("create_model_year_trend_chart")
    to retrieve it again.  Use the summary to describe findings to the user.

    Parameters
    ----------
    min_year : int, default 1990
        Earliest model year to include. Years before this value are filtered out.
    max_year : int | None, default None
        Latest model year to include. When None, all years up to the most recent
        in the database are included.

    Returns
    -------
    str
        JSON string with keys:
          "chart_name" : "create_model_year_trend_chart"
          "display"    : how the chart was opened
          "summary"    : {
              "min_year"               : int,
              "max_year"               : int | None,
              "total_complaints_in_range" : int,
              "year_range"             : {"min": int, "max": int},
              "top_5_years"            : [{"year": int, "count": int}, ...]
          }
    """
    chart_name = "create_model_year_trend_chart"

    import pandas as pd
    import numpy as np

    def _extract_year(value) -> int | None:
        # Mirror _extract_model_year from Create_Charts.
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            value = value.flat[0]
        elif isinstance(value, list):
            if not value:
                return None
            value = value[0]
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        year = int(value)
        return None if year == 9999 else year

    summary: dict = {
        "min_year": min_year,
        "max_year": max_year,
        "total_complaints_in_range": 0,
        "year_range": {},
        "top_5_years": [],
    }

    try:
        df = pd.read_parquet(PARQUET_PATH)
        years = df["YEARTXT"].apply(_extract_year).dropna().astype(int)
        years = years[years >= int(min_year)]
        if max_year is not None:
            years = years[years <= int(max_year)]

        summary["total_complaints_in_range"] = int(len(years))
        if not years.empty:
            counts = years.value_counts().sort_index()
            summary["year_range"] = {"min": int(counts.index.min()), "max": int(counts.index.max())}
            top5 = counts.sort_values(ascending=False).head(5)
            summary["top_5_years"] = [
                {"year": int(yr), "count": int(cnt)} for yr, cnt in top5.items()
            ]
    except Exception as e:
        summary["error"] = str(e)

    _LAST_CHART_DATA[chart_name] = summary

    # Headless / no matplotlib: return summary without rendering.
    if is_headless_mode() or not _CHART_BACKEND_READY:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    # --- local CLI: render and display -------------------------------------
    fig = create_model_year_trend_chart(
        str(PARQUET_PATH),
        min_year=min_year,
        max_year=max_year,
    )

    if fig is None:
        return json.dumps({
            "chart_name": chart_name,
            "rendered": False,
            "summary": summary,
        })

    png = _render_fig_to_png(fig)
    display_result = _display_png_bytes(png, chart_name)

    return json.dumps({
        "chart_name": chart_name,
        "rendered": True,
        "display": display_result,
        "summary": _LAST_CHART_DATA[chart_name],
    })


@tool
def describe_chart_data(chart_name: str) -> str:
    """
    Return the structured data summary for a previously rendered chart.

    This tool exists because the model CANNOT see rendered PNG images — it only
    receives the bytes size confirmation, not visual content.  After any chart
    tool runs, the model should call describe_chart_data() to retrieve the
    structured summary so it can discuss the chart meaningfully with the user.

    Valid chart_name values (must have been rendered first):
      - "create_bar_chart"
      - "create_model_year_chart"
      - "create_human_subsystem_frequency_chart"
      - "compare_LLM_to_NHTSA"
      - "create_subsystem_frequency_by_make_chart"
      - "create_subsystem_safety_signal_chart"
      - "create_model_year_trend_chart"

    Parameters
    ----------
    chart_name : str
        The name of the chart whose data you want to retrieve.  Must match
        the "chart_name" key returned by the corresponding chart tool.

    Returns
    -------
    str
        JSON string containing the structured summary dict for that chart,
        or {"error": "chart not yet rendered"} if the chart hasn't been run
        in this session.
    """
    return json.dumps(
        _LAST_CHART_DATA.get(chart_name, {"error": "chart not yet rendered"})
    )
