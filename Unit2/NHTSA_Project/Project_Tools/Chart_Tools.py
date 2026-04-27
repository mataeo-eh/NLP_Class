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

# ---------------------------------------------------------------------------
# CRITICAL: matplotlib backend MUST be set before any pyplot import.
# "Agg" is a non-interactive, off-screen rasteriser — it writes pixels to
# memory rather than opening a GUI window.  Setting it here ensures the entire
# LangGraph process uses Agg, which is required because we pipe rendered PNGs
# to Preview via stdin rather than calling plt.show().
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")          # must precede "import matplotlib.pyplot as plt"
import matplotlib.pyplot as plt  # noqa: E402  (import order is intentional)

import json
import os
import platform
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Ensure Create_Charts.py is importable: insert this file's directory at the
# front of sys.path so "import Create_Charts" resolves without needing the
# caller to manage sys.path.  We use index 0 so it takes priority over any
# stale package entries that might shadow local modules.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from Create_Charts import (  # noqa: E402
    create_bar_chart,
    create_model_year_chart,
    compare_LLM_to_NHTSA,
)

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

    # --- call underlying chart function ---
    # create_bar_chart(csv_path, columns, output_dir) → returns last fig or None
    # We pass OUTPUTS_DIR as output_dir so charts/ subfolder lands alongside
    # the CSV (mirrors existing Charts.ipynb usage).
    fig = create_bar_chart(str(csv_path), columns, str(OUTPUTS_DIR))

    # --- build structured summary so the LLM can reason about the chart ----
    # Load the CSV ourselves to compute per-column value_counts for the summary.
    # We do this independently of the chart function to keep the summary
    # deterministic even if the chart function's internals change.
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
                # Top 10 by frequency; convert int64 counts to plain int for JSON.
                vc = df[col].value_counts().head(10)
                summary["value_counts"][col] = {str(k): int(v) for k, v in vc.items()}
    except Exception as e:
        summary["error"] = str(e)

    _LAST_CHART_DATA[chart_name] = summary

    # --- render and display ------------------------------------------------
    if fig is None:
        # All columns were missing from the CSV; nothing was rendered.
        return json.dumps({
            "chart_name": chart_name,
            "display": "not rendered (no valid columns found)",
            "summary": summary,
        })

    png = _render_fig_to_png(fig)
    display_result = _display_png_bytes(png, chart_name)

    return json.dumps({
        "chart_name": chart_name,
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

    # --- build structured summary before rendering --------------------------
    # We load the parquet to compute the top-5 summary independently so the
    # LLM receives the data even if chart rendering later fails.
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

        # Build the same vehicle key used by the chart function.
        make_model = (
            df[make_col].astype(str).str.strip().str.upper()
            + " "
            + df[model_col].astype(str).str.strip().str.upper()
        )

        if include_year:
            def _extract_year_safe(val) -> str:
                # Mirror the logic in create_model_year_chart._extract_year.
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

    # --- call underlying chart function ------------------------------------
    # create_model_year_chart(df_path, *, include_year) → returns fig or None
    fig = create_model_year_chart(str(PARQUET_PATH), include_year=include_year)

    if fig is None:
        return json.dumps({
            "chart_name": chart_name,
            "display": "not rendered (data load failed)",
            "summary": summary,
        })

    png = _render_fig_to_png(fig)
    display_result = _display_png_bytes(png, chart_name)

    return json.dumps({
        "chart_name": chart_name,
        "display": display_result,
        "summary": _LAST_CHART_DATA[chart_name],
    })


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
    csv_path     = OUTPUTS_DIR / "Specific_Subsystem_Prompt.csv"

    # --- build structured summary before rendering --------------------------
    # Re-run the agreement classification logic independently so the LLM gets
    # a structured summary even if the chart function's display path changes.
    import ast
    import pandas as pd
    import numpy as np
    from collections import Counter

    summary: dict = {
        "total_complaints": 0,
        "agreement_counts": {"full": 0, "partial": 0, "none": 0},
        "percentages": {"full": 0.0, "partial": 0.0, "none": 0.0},
        "top_5_confused_pairs": [],
    }

    try:
        csv_df = pd.read_csv(csv_path)
        pq_df  = pd.read_parquet(PARQUET_PATH)

        def _normalize(labels):
            return frozenset(str(l).upper().strip() for l in labels)

        def _llm_matches(llm_label, nhtsa_set):
            for n in nhtsa_set:
                if llm_label == n:
                    return True
                if n.startswith(llm_label + ":") or n.startswith(llm_label + "/"):
                    return True
            return False

        full_agree = partial_agree = no_agree = 0
        miss_pairs: list[tuple[str, str]] = []

        for _, row in csv_df.iterrows():
            try:
                llm_labels = ast.literal_eval(row["subsystems"])
            except (ValueError, SyntaxError):
                llm_labels = []

            llm_set   = _normalize(llm_labels)
            nhtsa_raw = pq_df.loc[row["df_index"], "COMPDESC"]
            nhtsa_set = _normalize(list(nhtsa_raw))

            if llm_set == nhtsa_set:
                full_agree += 1
            else:
                any_match = any(_llm_matches(l, nhtsa_set) for l in llm_set)
                if any_match:
                    partial_agree += 1
                else:
                    no_agree += 1
                    for n in nhtsa_set:
                        for l in llm_set:
                            miss_pairs.append((n, l))

        total = full_agree + partial_agree + no_agree
        summary["total_complaints"] = total
        summary["agreement_counts"] = {
            "full": full_agree,
            "partial": partial_agree,
            "none": no_agree,
        }
        if total > 0:
            summary["percentages"] = {
                "full":    round(full_agree    / total * 100, 1),
                "partial": round(partial_agree / total * 100, 1),
                "none":    round(no_agree      / total * 100, 1),
            }

        # Top-5 most frequent (NHTSA label, LLM label) miss pairs.
        pair_counts = Counter(miss_pairs)
        top5_pairs = pair_counts.most_common(5)
        summary["top_5_confused_pairs"] = [
            {"nhtsa_label": n, "llm_label": l, "count": c}
            for (n, l), c in top5_pairs
        ]

    except Exception as e:
        summary["error"] = str(e)

    _LAST_CHART_DATA[chart_name] = summary

    # --- call underlying chart function ------------------------------------
    # compare_LLM_to_NHTSA(csv_path, parquet_path) → returns fig or None
    fig = compare_LLM_to_NHTSA(str(csv_path), str(PARQUET_PATH))

    if fig is None:
        return json.dumps({
            "chart_name": chart_name,
            "display": "not rendered (data load failed)",
            "summary": summary,
        })

    png = _render_fig_to_png(fig)
    display_result = _display_png_bytes(png, chart_name)

    return json.dumps({
        "chart_name": chart_name,
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
      - "compare_LLM_to_NHTSA"

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
