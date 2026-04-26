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
  filter_rows              — filter the complaint DB by column equality values

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
from Project_Tools.Audio_Playback import generate_TTS_audio
from Project_Tools.Audio_Capture import capture_and_transcribe

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

    If fields is provided, only those columns are included (unrecognised column names
    are silently ignored). Per-row NaN and empty-list values are dropped so the output
    stays compact. Non-serialisable types (Timestamps, pandas Int64, etc.) are
    stringified via json.dumps default=str.
    """
    if fields:
        valid_fields = [f for f in fields if f in df.columns]
        df = df[valid_fields]

    rows = []
    for _, row in df.iterrows():
        row_dict = {}
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


# ---------------------------------------------------------------------------
# Parquet tools
# ---------------------------------------------------------------------------

@tool
def get_rows_by_position(
    count: int = 1,
    indices: list | None = None,
    fields: list | None = None,
) -> str:
    """
    Fetch rows from the NHTSA complaints Parquet database by row position.

    Two modes:
      - indices provided: fetch those exact 0-based integer row positions.
        Use this when the user specifies a particular row number (e.g. "show me row 4587").
      - indices is None: randomly sample `count` rows from the database.
        Use this when the user asks for examples without specifying positions
        (e.g. "give me a few sample complaints").

    Parameters
    ----------
    count : int
        Number of rows to randomly sample when indices is not provided. Capped at 10.
    indices : list of int, optional
        Explicit 0-based row positions to fetch. If provided, count is ignored.
        Capped at 10 entries.
    fields : list of str, optional
        Column names to include in the result. If None, all non-NaN columns
        are returned for each row.

    Returns
    -------
    str
        JSON array of row objects. Each object contains only non-NaN field values.
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

    result_df = df.iloc[indices].reset_index(drop=True)
    del df  # release full DataFrame from memory before returning

    # fields already applied via column projection above
    return _rows_to_json(result_df, fields=None)


@tool
def filter_rows(
    filters: dict,
    limit: int = 5,
    fields: list | None = None,
) -> str:
    """
    Filter the NHTSA complaints Parquet database by column equality values.

    Supports all 49 complaint columns. The COMPDESC column is list-typed (a single
    complaint can span multiple components after deduplication). Filtering on COMPDESC
    checks whether the value appears anywhere in that list, e.g.
    filters={"COMPDESC": "ENGINE COOLING SYSTEM"} returns complaints where
    ENGINE COOLING SYSTEM is one of the listed component categories.

    Parameters
    ----------
    filters : dict
        Column-to-value equality filters. Values can be strings or numbers. Examples:
          {"MAKETXT": "TOYOTA", "CRASH": "Y"}
          {"COMPDESC": "SERVICE BRAKES", "YEARTXT": 2023}
          {"FIRE": "Y", "DEATHS": 1}
    limit : int
        Maximum number of matching rows to return. Capped at 10.
    fields : list of str, optional
        Columns to include in each result row. If None, all non-NaN columns returned.

    Returns
    -------
    str
        JSON array of matching row objects, or a message if no matches were found.
    """
    # Load only the columns needed: filter columns + result fields.
    # When fields=None, load everything since we don't know which columns are wanted.
    filter_cols = list(filters.keys())
    if fields:
        cols_to_load = list(set(filter_cols) | set(fields))
    else:
        cols_to_load = None

    df = pd.read_parquet(PARQUET_PATH, columns=cols_to_load)

    # Apply each equality filter in sequence, short-circuiting on empty result
    for col, val in filters.items():
        df = _apply_filter(df, col, val)
        if df.empty:
            del df
            return json.dumps({"result": "No matching rows found for the given filters."})

    cap = min(limit, MAX_ROWS)
    result_df = df.head(cap).reset_index(drop=True)
    del df

    if result_df.empty:
        return json.dumps({"result": "No matching rows found for the given filters."})

    return _rows_to_json(result_df, fields)


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

    result_df = pd.concat(matched_chunks).head(cap).reset_index(drop=True)
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
    # Speak the question aloud via TTS so the user hears it through their speakers.
    generate_TTS_audio(
        text=question,
        model="mlx-community/Kokoro-82M-bf16",
        voice="af_sky",
        speed=0.85,
        lang_code="a",
        play=True,
        streaming_interval=0.2,
        stream=True,
        save=False,
    )
    return "Question asked. Call User_Answer() to retrieve the response."


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
    # Capture the user's spoken reply, transcribe it via Whisper, and return
    # the transcript string. Blocks until end-of-speech is detected.
    return capture_and_transcribe()
