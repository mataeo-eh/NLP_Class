"""
NHTSA_Query_Tools.py
--------------------
LangChain-compatible tools for the retrieve_data node to query:
  1. The NHTSA complaints Parquet database (NHTSA/complaints_cleaned.parquet)
  2. Pipeline output CSV files (Outputs/*.csv)

Each tool loads data on demand and releases it after the call — no module-level
DataFrame caches — to keep RAM usage low alongside the TTS model.

Tool catalogue
--------------
Parquet tools:
  get_rows_by_position     — fetch rows by explicit positions or random sample
  list_nhtsa_component_categories
                           — list the top-level COMPDESC component categories available for filtering
  filter_rows              — fetch a small preview of matching complaint rows
  get_recent_complaints    — fetch the newest complaints matching filters
  count_complaints         — exact full-dataset row count after filtering
  group_complaints         — exact full-dataset counts grouped by complaint columns
  summarize_complaints     — exact descriptive statistics for complaint columns
  build_complaint_stats_dataset
                           — exact filtered table stored server-side for RMCP stats tools

CSV tools:
  list_csv_files           — list available CSV output files in Outputs/
  get_csv_schema           — use mercury LLM to describe a CSV file's columns
  filter_csv               — filter a CSV output file by column equality values
  get_csv_rows_by_position — fetch CSV rows by explicit positions or random sample

Clarification tools (audio pipeline):
  Ask_User                 — speak a clarifying question to the user via TTS
  User_Answer              — capture the user's spoken reply and return the transcript
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Import mercury_llm from LangGraph/config.py for get_csv_schema
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "LangGraph"))
from LangGraph.config import mercury_llm

# ---------------------------------------------------------------------------
# Import audio pipeline — TTS for Ask_User, STT for User_Answer
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from Project_Tools.Runtime_Options import (
    HostedUserInteractionRequired,
    consume_headless_clarification_response,
    is_audio_enabled,
    is_headless_mode,
)
from LLM_Tools.MCP_To_Tools import register_stats_dataset
from Prompts import NHTSA_COMPONENT_TAXONOMY

# ---------------------------------------------------------------------------
# File path constants
# ---------------------------------------------------------------------------

# Parquet database built by NHTSA/Build_DF.py
PARQUET_PATH = Path(__file__).parent.parent / "NHTSA" / "complaints_cleaned.parquet"

# Directory where pipeline CSV output files are written
OUTPUTS_DIR = Path(__file__).parent.parent / "Outputs"

# Hard cap on rows returned by any single tool call
MAX_ROWS = 10


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Python list and numpy.ndarray are both treated as "list-like" cell containers.
# Build_DF.py writes Python lists into deduplicated columns, but pyarrow rehydrates
# them as numpy.ndarray on read, so every helper that inspects cell type must
# accept both.
_LIST_LIKE = (list, np.ndarray)


def _is_listlike(v) -> bool:
    """True for Python list or numpy.ndarray cells."""
    return isinstance(v, _LIST_LIKE)


def _is_nan(v) -> bool:
    """Return True for any flavor of missing value, safely handling list-typed cells."""
    if v is None:
        return True
    if _is_listlike(v):
        # Empty list / empty ndarray represents a null after Build_DF._normalize_list_columns.
        return len(v) == 0
    try:
        return pd.isna(v)
    except (TypeError, ValueError):
        return False


def _rows_to_json(df: pd.DataFrame, fields: list | None) -> str:
    """
    Convert a DataFrame to a compact JSON string.

    Each output row dict starts with an "index" field that holds the row's
    position in the original data source (parquet position or CSV row number).
    This is taken directly from df.index, so callers MUST preserve the original
    index in df — do NOT call reset_index(drop=True) before passing the df here.

    If fields is provided, only those columns are included (unrecognised column names
    are silently ignored). Per-row NaN and empty-list values are dropped so the output
    stays compact. Non-serialisable types (Timestamps, pandas Int64, etc.) are
    stringified via json.dumps default=str.
    """
    if fields:
        valid_fields = [f for f in fields if f in df.columns]
        df = df[valid_fields]

    rows = []
    for idx, row in df.iterrows():
        # "index" is injected first so it appears at the top of each row dict in
        # the JSON output. Cast to plain int because pandas indices are often
        # numpy.int64, which serializes fine but reads cleaner as a Python int.
        row_dict = {"index": int(idx)}
        for k, v in row.items():
            if _is_nan(v):
                continue
            # Convert ndarray cells (parquet-rehydrated lists) to plain lists so the
            # JSON output is a real array rather than the str repr of an ndarray.
            if isinstance(v, np.ndarray):
                v = v.tolist()
            row_dict[k] = v
        rows.append(row_dict)

    return json.dumps(rows, default=str, indent=2)


def _apply_filter(df: pd.DataFrame, col: str, val) -> pd.DataFrame:
    """
    Apply a single equality filter to a DataFrame column.

    Handles list-typed columns (e.g. COMPDESC, MAKETXT, MFR_NAME after deduplication
    in Build_DF.py) by checking whether val is a member of the cell rather than
    requiring exact equality. For scalar columns, performs a standard equality
    comparison. Silently ignores filter columns that do not exist in the DataFrame.

    NOTE: parquet (pyarrow) rehydrates Python lists as numpy.ndarray, so we use
    _is_listlike() rather than isinstance(..., list) — otherwise comparing
    `ndarray == scalar` returns an elementwise boolean array per cell, which
    pandas cannot reduce to a row mask and raises:
        ValueError: The truth value of an array with more than one element is ambiguous.
    """
    if col not in df.columns:
        return df

    # Detect list-typed columns by probing the first cell that is neither NaN nor empty.
    # An empty ndarray is also list-like, so it would also send us down the list branch,
    # but we prefer a non-empty probe so we never mis-classify a column whose first row
    # happens to be a stray empty array.
    probe = None
    for v in df[col]:
        if _is_nan(v):
            continue
        probe = v
        break

    if probe is not None and _is_listlike(probe):
        # List column — check membership rather than equality.
        # val may itself be a list (LLM passing multiple values), so accept any match.
        if isinstance(val, list):
            return df[df[col].apply(
                lambda x: any(v in x for v in val) if _is_listlike(x) else x in val
            )]
        return df[df[col].apply(
            lambda x: val in x if _is_listlike(x) else x == val
        )]

    # Scalar column — use isin() when val is a list so pandas gets a proper boolean mask
    if isinstance(val, list):
        return df[df[col].isin(val)]
    return df[df[col] == val]


def _apply_filters(df: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """Apply zero or more complaint-column filters in sequence."""
    filters = filters or {}
    for col, val in filters.items():
        df = _apply_filter(df, col, val)
        if df.empty:
            break
    return df


def _to_json_safe(value):
    """Convert pandas/numpy scalars into plain JSON-friendly Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _to_stats_cell(value):
    """
    Convert one DataFrame cell into a scalar value safe for RMCP table schemas.

    RMCP statistical tools expect column-wise arrays of scalar JSON values.
    Complaint parquet columns can contain list-like cells, so those are encoded
    as compact JSON strings instead of nested arrays-of-arrays.
    """
    safe_value = _to_json_safe(value)
    if isinstance(safe_value, list):
        return json.dumps(safe_value, default=str)
    return safe_value


def _series_is_listlike(series: pd.Series) -> bool:
    """True when the first non-missing cell in the series is list-like."""
    for value in series:
        if _is_nan(value):
            continue
        return _is_listlike(value)
    return False


def _explode_series_values(series: pd.Series, include_null: bool = False) -> pd.Series:
    """Flatten scalar or list-like cell values into a single 1-D object series."""
    values = []
    for cell in series:
        if _is_nan(cell):
            if include_null:
                values.append(None)
            continue

        if _is_listlike(cell):
            iterable = cell.tolist() if isinstance(cell, np.ndarray) else list(cell)
            if not iterable:
                if include_null:
                    values.append(None)
                continue
            for item in iterable:
                if _is_nan(item):
                    if include_null:
                        values.append(None)
                    continue
                values.append(_to_json_safe(item))
            continue

        values.append(_to_json_safe(cell))

    return pd.Series(values, dtype="object")


def _build_numeric_summary(series: pd.Series) -> dict:
    """Return a JSON-friendly numeric describe() summary for one series."""
    described = series.describe(percentiles=[0.25, 0.5, 0.75])
    summary = {"kind": "numeric"}
    for key, value in described.items():
        summary[str(key)] = _to_json_safe(value)
    return summary


def _build_categorical_summary(series: pd.Series) -> dict:
    """Return a JSON-friendly categorical describe() summary for one series."""
    described = series.astype("object").describe()
    summary = {"kind": "categorical"}
    for key, value in described.items():
        summary[str(key)] = _to_json_safe(value)
    return summary


# ---------------------------------------------------------------------------
# Parquet tools
# ---------------------------------------------------------------------------

@tool
def list_nhtsa_component_categories() -> str:
    """Return the available top-level NHTSA component categories for COMPDESC filtering.

    Use this before fetching complaint rows when the user names a subsystem or
    component. The returned component names are the exact top-level taxonomy
    values you should use in filters={"COMPDESC": "<component>"}.

    If the user's wording does not exactly match one category, choose the
    closest category from this list before calling filter_rows or
    get_recent_complaints. COMPDESC filtering checks whether that chosen
    category appears anywhere in the complaint's component list.
    """
    return json.dumps(
        {
            "filter_column": "COMPDESC",
            "count": len(NHTSA_COMPONENT_TAXONOMY),
            "components": NHTSA_COMPONENT_TAXONOMY,
            "usage_note": (
                "Choose the closest category from this list, then pass it as "
                'filters={"COMPDESC": "<CATEGORY>"} to filter_rows or '
                "get_recent_complaints."
            ),
        },
        indent=2,
    )


@tool
def get_rows_by_position(
    count: int = 1,
    indices: list | None = None,
    fields: list | None = None,
) -> str:
    """Fetch complaint rows by exact index or random sample.

    Use this to read complaint text or inspect specific examples. This is a row
    preview tool, not a whole-dataset statistics tool. Results are capped at 10 rows.
    """
    # Use column projection when specific fields are requested to reduce RAM load.
    # When fields=None, all 49 columns are returned, so no projection is applied.
    cols_to_load = list(fields) if fields else None
    df = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)
    n_rows = len(df)

    if indices is not None:
        # Explicit positions — cap and clamp to valid range
        indices = [i for i in indices[:MAX_ROWS] if 0 <= i < n_rows]
    else:
        # Random sample without replacement
        sample_size = min(count, MAX_ROWS, n_rows)
        indices = random.sample(range(n_rows), sample_size)

    # NOTE: do NOT call reset_index(drop=True) here. _rows_to_json reads
    # df.index to populate the "index" field on every row, which downstream
    # nodes (analyze, csv writers) rely on to track the original parquet
    # position of each complaint.
    result_df = df.iloc[indices]
    del df  # release full DataFrame from memory before returning

    # fields already applied via column projection above
    return _rows_to_json(result_df, fields=None)


@tool
def filter_rows(
    filters: dict,
    limit: int = 5,
    fields: list | None = None,
) -> str:
    """Return a small preview of complaint rows matching equality filters.

    Use this when you need example complaints or complaint text after applying
    filters. This tool only returns the first few matching rows and must not be
    used to infer exact whole-dataset counts, modes, or distributions.
    """
    # Load only the columns needed: filter columns + result fields.
    # When fields=None, load everything since we don't know which columns are wanted.
    filter_cols = list(filters.keys())
    if fields:
        cols_to_load = list(set(filter_cols) | set(fields))
    else:
        cols_to_load = None

    df = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)

    df = _apply_filters(df, filters)
    if df.empty:
        del df
        return json.dumps({"result": "No matching rows found for the given filters."})

    cap = min(limit, MAX_ROWS)
    # Preserve df.index so _rows_to_json can attach the original parquet row
    # position to each output dict. .head(cap) keeps the index intact.
    result_df = df.head(cap)
    del df

    if result_df.empty:
        return json.dumps({"result": "No matching rows found for the given filters."})

    return _rows_to_json(result_df, fields)


@tool
def get_recent_complaints(
    filters: dict | None = None,
    limit: int = 5,
    fields: list | None = None,
    date_field: str = "DATEA",
) -> str:
    """Return the newest complaint rows matching filters, sorted by one date column.

    Use this when the user asks for the latest, newest, or most recent
    complaints. Unlike filter_rows, this tool explicitly sorts by a complaint
    date field before applying the row cap, so it can answer recency-based
    requests deterministically instead of relying on parquet row order.
    """
    filters = filters or {}
    cols_to_load = list(dict.fromkeys(list(filters.keys()) + ([date_field] if date_field else []) + (fields or [])))
    df = pd.read_parquet(PARQUET_PATH, columns=cols_to_load or None)

    if date_field not in df.columns:
        del df
        return json.dumps(
            {
                "error": (
                    f"date_field {date_field!r} is not present in the complaint parquet file. "
                    "Choose one of the available date columns from the schema summary."
                )
            },
            indent=2,
        )

    filtered = _apply_filters(df, filters)
    del df
    if filtered.empty:
        del filtered
        return json.dumps({"result": "No matching rows found for the given filters."})

    # DATEA and related complaint date columns are stored as strings in the
    # cleaned parquet file. Parse them explicitly so "most recent" sorts by
    # real chronology rather than lexicographic string order.
    sortable = filtered.assign(
        __parsed_sort_date=pd.to_datetime(filtered[date_field], errors="coerce")
    )
    cap = min(limit, MAX_ROWS)
    result_df = (
        sortable
        .sort_values("__parsed_sort_date", ascending=False, na_position="last")
        .head(cap)
        .drop(columns=["__parsed_sort_date"])
    )
    del filtered
    del sortable

    if result_df.empty:
        return json.dumps({"result": "No matching rows found for the given filters."})

    return _rows_to_json(result_df, fields)


@tool
def count_complaints(filters: dict | None = None) -> str:
    """Return the exact number of complaints matching filters across the full parquet database.

    Use this for exact whole-dataset counts. Unlike filter_rows, this scans all
    matching complaints and returns a count instead of sample rows.
    """
    filters = filters or {}
    cols_to_load = list(filters.keys()) if filters else None
    df = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)
    total_rows = int(len(df))
    filtered = _apply_filters(df, filters)
    matching_rows = int(len(filtered))
    del df
    del filtered

    share = (matching_rows / total_rows) if total_rows else 0.0
    return json.dumps(
        {
            "filters": filters,
            "matching_rows": matching_rows,
            "total_rows": total_rows,
            "share_of_dataset": round(share, 6),
        },
        default=str,
        indent=2,
    )


@tool
def group_complaints(
    group_by: list[str],
    filters: dict | None = None,
    top_n: int = 25,
    include_null: bool = False,
    ascending: bool = False,
) -> str:
    """Return exact full-dataset counts grouped by complaint columns.

    Use this for questions like "what is the most common year/state/make among
    all engine complaints?" Unlike filter_rows, this computes exact aggregate
    counts over all matching complaints. Single list-like columns such as
    COMPDESC are exploded before counting.
    """
    filters = filters or {}
    group_by = [col for col in dict.fromkeys(group_by or []) if col]
    if not group_by:
        return json.dumps({"error": "group_by must contain at least one complaint column name."})

    top_n = max(1, min(int(top_n), 100))
    cols_to_load = list(dict.fromkeys(list(filters.keys()) + group_by))
    df = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)
    filtered = _apply_filters(df, filters)
    matching_rows = int(len(filtered))

    missing = [col for col in group_by if col not in filtered.columns]
    if missing:
        del df
        del filtered
        return json.dumps(
            {
                "error": "One or more group_by columns were not found.",
                "missing_columns": missing,
            },
            indent=2,
        )

    if matching_rows == 0:
        del df
        del filtered
        return json.dumps(
            {
                "filters": filters,
                "group_by": group_by,
                "matching_rows": 0,
                "distinct_groups": 0,
                "groups": [],
            },
            indent=2,
        )

    listlike_group_cols = [col for col in group_by if _series_is_listlike(filtered[col])]
    if listlike_group_cols and len(group_by) > 1:
        del df
        del filtered
        return json.dumps(
            {
                "error": (
                    "group_complaints only supports list-like grouping when exactly one "
                    "group_by column is provided."
                ),
                "listlike_columns": listlike_group_cols,
            },
            indent=2,
        )

    if listlike_group_cols:
        column = group_by[0]
        exploded = _explode_series_values(filtered[column], include_null=include_null)
        counts = exploded.value_counts(dropna=not include_null, ascending=ascending)
        groups = [
            {"value": _to_json_safe(value), "row_count": int(count)}
            for value, count in counts.head(top_n).items()
        ]
        distinct_groups = int(len(counts))
        del exploded
    else:
        grouped = filtered[group_by]
        if not include_null:
            grouped = grouped.dropna(subset=group_by)

        counts_df = (
            grouped.groupby(group_by, dropna=not include_null)
            .size()
            .reset_index(name="row_count")
        )
        counts_df = counts_df.sort_values(
            by=["row_count"] + group_by,
            ascending=[ascending] + [True] * len(group_by),
            kind="stable",
        )
        distinct_groups = int(len(counts_df))
        groups = []
        for _, row in counts_df.head(top_n).iterrows():
            entry = {col: _to_json_safe(row[col]) for col in group_by}
            entry["row_count"] = int(row["row_count"])
            groups.append(entry)
        del grouped
        del counts_df

    del df
    del filtered
    return json.dumps(
        {
            "filters": filters,
            "group_by": group_by,
            "matching_rows": matching_rows,
            "distinct_groups": distinct_groups,
            "top_group": groups[0] if groups else None,
            "groups": groups,
        },
        default=str,
        indent=2,
    )


@tool
def summarize_complaints(
    columns: list[str],
    filters: dict | None = None,
    include_null: bool = False,
) -> str:
    """Return exact descriptive statistics for complaint columns over all matching rows.

    Use this for dataset-wide numeric summaries or categorical modes after
    filtering. Unlike filter_rows, this summarizes all matching complaints
    instead of returning example rows.
    """
    filters = filters or {}
    columns = [col for col in dict.fromkeys(columns or []) if col]
    if not columns:
        return json.dumps({"error": "columns must contain at least one complaint column name."})

    cols_to_load = list(dict.fromkeys(list(filters.keys()) + columns))
    df = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)
    filtered = _apply_filters(df, filters)
    matching_rows = int(len(filtered))

    summaries: dict[str, dict] = {}
    warnings: list[str] = []

    for column in columns:
        if column not in filtered.columns:
            warnings.append(f"Column '{column}' was not found and was skipped.")
            continue

        series = filtered[column]
        if _series_is_listlike(series):
            exploded = _explode_series_values(series, include_null=include_null)
            counts = exploded.value_counts(dropna=not include_null)
            summaries[column] = {
                "kind": "list_like",
                "count": int(len(exploded)),
                "distinct_values": int(len(counts)),
                "top_values": [
                    {"value": _to_json_safe(value), "count": int(count)}
                    for value, count in counts.head(10).items()
                ],
            }
            del exploded
            continue

        non_missing_mask = series.apply(lambda value: not _is_nan(value))
        clean = series[non_missing_mask]
        if clean.empty:
            summaries[column] = {"kind": "empty", "count": 0}
            continue

        numeric = pd.to_numeric(clean, errors="coerce")
        if numeric.notna().all():
            summaries[column] = _build_numeric_summary(numeric)
            continue

        summaries[column] = _build_categorical_summary(clean)

    del df
    del filtered
    payload = {
        "filters": filters,
        "matching_rows": matching_rows,
        "column_summaries": summaries,
    }
    if warnings:
        payload["warnings"] = warnings
    return json.dumps(payload, default=str, indent=2)


@tool
def build_complaint_stats_dataset(
    columns: list[str],
    filters: dict | None = None,
    max_rows: int = 5000,
    include_index: bool = True,
) -> str:
    """
    Store an exact filtered parquet table server-side for downstream RMCP stats tools.

    The tool returns a compact dataset manifest with `dataset_id`. The raw rows
    stay on the backend so later statistical tool calls can reference them
    without forcing the model to copy a full table back through prompt context.
    """
    if not columns:
        return json.dumps(
            {"error": "columns must contain at least one complaint column."},
            indent=2,
        )

    try:
        max_rows = int(max_rows)
    except (TypeError, ValueError):
        return json.dumps({"error": "max_rows must be an integer."}, indent=2)

    max_rows = max(1, min(max_rows, 50000))
    filters = filters or {}
    requested_columns = [col for col in dict.fromkeys(columns) if col]

    pq_df = pd.read_parquet(PARQUET_PATH)
    filtered = _apply_filters(pq_df, filters)
    row_count = int(len(filtered))

    missing_columns = [col for col in requested_columns if col not in filtered.columns]
    if missing_columns:
        return json.dumps(
            {
                "error": "One or more requested complaint columns were not found.",
                "missing_columns": missing_columns,
            },
            indent=2,
        )

    if row_count > max_rows:
        return json.dumps(
            {
                "error": (
                    "The exact filtered table is larger than max_rows. Narrow the filters "
                    "or raise max_rows if you intentionally want a larger stats dataset."
                ),
                "row_count": row_count,
                "max_rows": max_rows,
                "filters": filters,
                "columns": requested_columns,
            },
            indent=2,
        )

    data: dict[str, list] = {}
    if include_index:
        data["df_index"] = [int(idx) for idx in filtered.index.tolist()]

    for column in requested_columns:
        data[column] = [_to_stats_cell(value) for value in filtered[column].tolist()]

    manifest = register_stats_dataset(
        data=data,
        source="complaints_parquet",
        description="Exact filtered complaint dataset prepared for hosted statistical tools.",
        metadata={
            "filters": filters,
            "requested_columns": requested_columns,
            "include_index": bool(include_index),
        },
    )
    return json.dumps(manifest, default=str, indent=2)


# ---------------------------------------------------------------------------
# CSV tools
# ---------------------------------------------------------------------------

@tool
def list_csv_files() -> str:
    """
    List all CSV output files available in the Outputs directory.

    Use this first when the user asks about pipeline output data such as LLM safety
    labels, subsystem classifications, or reasoning outputs. The filenames describe
    their contents — use them together with the user's query to decide which file
    to open next.

    Returns
    -------
    str
        JSON object with a "csv_files" key listing the available CSV filenames.
    """
    csv_files = [f for f in os.listdir(OUTPUTS_DIR) if f.endswith(".csv")]

    if not csv_files:
        return json.dumps({"result": "No CSV files found in the Outputs directory."})

    return json.dumps({"csv_files": sorted(csv_files)})


@tool
def get_csv_schema(filename: str) -> str:
    """
    Generate a schema description for a CSV output file using the mercury LLM.

    Reads the first 50 rows of the file and sends column names plus up to 10 sample
    values per column to mercury, which returns a structured description of what each
    column contains and its apparent data type.

    Always call this before querying a CSV file you have not yet inspected, so you
    know which columns are available for filtering and what values they hold.

    Parameters
    ----------
    filename : str
        Name of the CSV file in the Outputs directory (filename only, e.g.
        "safety_labels.csv" — not a full path).

    Returns
    -------
    str
        Mercury's schema description of the file's columns and content.
    """
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return json.dumps({"error": f"File '{filename}' not found in the Outputs directory."})

    # Read only a small sample — enough to get representative values per column
    # without pulling a potentially large file into memory.
    sample_df = pd.read_csv(path, nrows=50)

    # Build a structured column summary: column name -> list of up to 10 sample values
    column_samples = {}
    for col in sample_df.columns:
        samples = sample_df[col].dropna().head(10).tolist()
        column_samples[col] = samples

    del sample_df  # release sample before calling LLM

    prompt = (
        f"Analyze the following CSV file column data and generate a concise schema description.\n\n"
        f"File: {filename}\n\n"
        f"For each column, describe:\n"
        f"  1. What the column represents (its semantic meaning in the context of NHTSA vehicle safety analysis)\n"
        f"  2. The apparent data type (text, integer, float, boolean, date, categorical list, etc.)\n"
        f"  3. Notable patterns, value ranges, or coding conventions visible in the samples\n\n"
        f"Column data (column_name: [up to 10 sample values]):\n"
        f"{json.dumps(column_samples, default=str, indent=2)}\n\n"
        f"Return a clear, structured schema description that allows another AI system to write "
        f"accurate queries against this file — specifically which columns are useful for "
        f"filtering and which columns contain the primary result or output data."
    )

    response = mercury_llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def filter_csv(
    filename: str,
    filters: dict,
    limit: int = 5,
    fields: list | None = None,
) -> str:
    """
    Filter a CSV output file by column equality values and return matching rows.

    Reads the file in chunks to avoid loading it fully into RAM. Stops reading
    once enough matching rows have been found.

    Call get_csv_schema first if you are not yet familiar with the file's columns.

    Parameters
    ----------
    filename : str
        Name of the CSV file in the Outputs directory (e.g. "safety_labels.csv").
    filters : dict
        Column-to-value equality filters, e.g.:
          {"urgency": "Emergent", "make": "FORD"}
          {"llm_label": "High", "human_label": "Low"}
    limit : int
        Maximum number of matching rows to return. Capped at 10.
    fields : list of str, optional
        Columns to include in each result row. If None, all non-NaN columns returned.

    Returns
    -------
    str
        JSON array of matching row objects, or a message if no matches were found.
    """
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return json.dumps({"error": f"File '{filename}' not found in the Outputs directory."})

    cap = min(limit, MAX_ROWS)
    matched_chunks = []
    total_found = 0

    # Read in 500-row chunks and stop as soon as we have enough results.
    # This avoids loading the full file into memory for large CSV outputs.
    for chunk in pd.read_csv(path, chunksize=500):
        filtered = chunk.copy()

        for col, val in filters.items():
            filtered = _apply_filter(filtered, col, val)

        if not filtered.empty:
            matched_chunks.append(filtered)
            total_found += len(filtered)

        if total_found >= cap:
            break

    if not matched_chunks:
        return json.dumps({"result": "No matching rows found for the given filters."})

    # Preserve the chunk indices so each row dict carries its original CSV row
    # position. pd.read_csv with chunksize gives each chunk an index that
    # represents its global row position in the file (chunk 0: 0..499,
    # chunk 1: 500..999, etc.), which survives _apply_filter and concat.
    result_df = pd.concat(matched_chunks).head(cap)
    return _rows_to_json(result_df, fields)


@tool
def get_csv_rows_by_position(
    filename: str,
    count: int = 1,
    indices: list | None = None,
    fields: list | None = None,
) -> str:
    """
    Fetch rows from a CSV output file by row position.

    Two modes — mirrors get_rows_by_position for the Parquet database:
      - indices provided: fetch those exact 0-based row positions.
      - indices is None: randomly sample `count` rows from the file.

    Parameters
    ----------
    filename : str
        Name of the CSV file in the Outputs directory (e.g. "safety_labels.csv").
    count : int
        Number of rows to randomly sample when indices is not provided. Capped at 10.
    indices : list of int, optional
        Explicit 0-based row positions to fetch. Capped at 10 entries.
    fields : list of str, optional
        Columns to include in each result row. If None, all non-NaN columns returned.

    Returns
    -------
    str
        JSON array of row objects.
    """
    path = OUTPUTS_DIR / filename
    if not path.exists():
        return json.dumps({"error": f"File '{filename}' not found in the Outputs directory."})

    # Read only the first column to count total rows cheaply, without loading the full file.
    count_df = pd.read_csv(path, usecols=[0])
    n_rows = len(count_df)
    del count_df

    if indices is not None:
        indices = [i for i in indices[:MAX_ROWS] if 0 <= i < n_rows]
    else:
        sample_size = min(count, MAX_ROWS, n_rows)
        indices = random.sample(range(n_rows), sample_size)

    rows_to_keep = set(indices)

    # Use a skiprows callable to read only the target rows in a single pass.
    # i=0 is the header row (always kept). For data rows, i-1 gives the 0-based index.
    df = pd.read_csv(
        path,
        skiprows=lambda i: i > 0 and (i - 1) not in rows_to_keep,
        usecols=fields if fields else None,
    )

    # pd.read_csv with skiprows produces a fresh RangeIndex (0..N-1) rather than
    # preserving the original CSV positions. The kept rows arrive in the file's
    # natural order, so we reattach the original indices by sorting our request
    # list and assigning it back as the df.index. _rows_to_json then surfaces
    # those positions in the "index" field of each output row.
    df.index = pd.Index(sorted(rows_to_keep))

    return _rows_to_json(df, fields=None)  # fields already projected above


# ---------------------------------------------------------------------------
# Clarification stubs
# ---------------------------------------------------------------------------

@tool
def Ask_User(question: str) -> str:
    """
    Pose a clarifying question to the user before executing a data fetch.

    Use this when the user's request is genuinely ambiguous in a way that would
    cause a wrong or unhelpful result without clarification. Common cases:
      - Result count is unclear and the answer could reasonably be 1 or several.
      - Desired fields are ambiguous between a targeted single-field lookup and
        a full record retrieval.

    After calling this tool, call User_Answer() to retrieve the user's response
    before proceeding with the query.

    Parameters
    ----------
    question : str
        The clarifying question to pose to the user.

    Returns
    -------
    str
        Confirmation that the question was asked. Call User_Answer() next.
    """
    # Hosted backend: there is a real human in the browser, but this Python
    # process cannot ask them synchronously. Raise a structured pause signal so
    # the FastAPI adapter can emit the question to the browser, collect the
    # answer there, and resume later. Ask_User is the first half of the old
    # two-tool contract, so on resume the model still needs to call
    # User_Answer() to consume the stored reply.
    if is_headless_mode():
        raise HostedUserInteractionRequired(
            question,
            result_mode="confirm_then_answer",
        )

    # In audio mode, speak the question aloud via TTS so the user hears it through
    # their speakers. In text mode, print the same question and let stdin carry
    # the follow-up response through User_Answer().
    if not is_audio_enabled():
        print(f"[Ask_User] {question}")
        return "Question asked in text mode. Call User_Answer() to retrieve the response."

    try:
        from Project_Tools.Audio_Playback import generate_TTS_audio

        # Neither `model=` nor `voice=` is passed: generate_TTS_audio
        # dispatches on the runtime voice-model selector and voice-preset set
        # at CLI parse time (--voice-model and --voice-preset). Each engine's
        # branch routes the runtime preset to the right underlying parameter
        # (kokoro voice=, cartesia voice_id=, deepgram model_id=). The
        # kokoro-shaped lang_code / streaming_interval / stream args are read
        # only on the kokoro branch and silently ignored on cartesia / deepgram.
        generate_TTS_audio(
            text=question,
            speed=0.85,
            lang_code="a",
            play=True,
            streaming_interval=0.2,
            stream=True,
            save=False,
        )
        return "Question asked. Call User_Answer() to retrieve the response."
    except Exception as exc:
        return f"[audio unavailable: {exc}] Question text: {question}"


@tool
def User_Answer() -> str:
    """
    Retrieve the user's response to the most recently asked clarifying question.

    Always call this immediately after Ask_User() — never call it before a question
    has been asked. Use the returned string to refine your retrieval parameters
    before executing the data fetch.

    Returns
    -------
    str
        The user's plain-text response.
    """
    if is_headless_mode():
        hosted_answer = consume_headless_clarification_response()
        if hosted_answer is not None:
            return hosted_answer
        return (
            "[User_Answer unavailable: the hosted clarification response was not "
            "seeded before this tool call. Ask a new clarification question or "
            "continue without the missing answer.]"
        )
    # Capture the user's spoken reply in audio mode, or collect a typed reply in
    # text mode. Both branches return a plain-text string so the calling node
    # does not need mode-specific logic.
    if not is_audio_enabled():
        return input("Your response: ").strip()

    try:
        from Project_Tools.Audio_Capture import capture_and_transcribe

        return capture_and_transcribe()
    except Exception as exc:
        return f"[audio unavailable: {exc}]"
