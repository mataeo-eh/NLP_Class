import json
import re
import csv
import os
import threading
import time
import random

# litellm is only consulted by process_and_store_response (the retry wrapper used
# by NHTSA_Project_Main.py's CLI). The LangGraph pipeline imports this module
# solely for append_to_csv / ensure_output_dir / get_processed_df_indices, none
# of which touch litellm. The FastAPI backend on Render therefore doesn't need
# litellm installed — wrapping the import in try/except keeps this module
# importable in that environment. If litellm IS unavailable and someone calls
# process_and_store_response anyway, _RETRYABLE_ERRORS is empty so server-side
# errors propagate without retry rather than crashing on a missing attribute.
try:
    import litellm
except ImportError:
    litellm = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Per-CSV write locks
# ---------------------------------------------------------------------------
# Each output CSV path gets its own lock. Workers writing to different CSVs
# never block each other. Workers writing to the same CSV are serialized only
# for the instant the row is appended — far shorter than any LLM call.
_csv_locks: dict = {}
_csv_locks_guard = threading.Lock()  # protects the _csv_locks dict itself


def _get_csv_lock(csv_path: str) -> threading.Lock:
    """Return the dedicated Lock for a given CSV path, creating it on first access."""
    with _csv_locks_guard:
        if csv_path not in _csv_locks:
            _csv_locks[csv_path] = threading.Lock()
        return _csv_locks[csv_path]


# ---------------------------------------------------------------------------
# Server-side error retry config
# ---------------------------------------------------------------------------
# These are transient infrastructure errors — the request was well-formed but
# the server couldn't handle it. Each is worth retrying after a back-off.
_RETRYABLE_ERRORS: tuple = (
    (
        litellm.RateLimitError,          # 429 — rate limit hit
        litellm.ServiceUnavailableError, # 503 — server temporarily unavailable
        litellm.Timeout,                 # request timed out before a response
        litellm.APIConnectionError,      # network-level connection failure
        litellm.InternalServerError,     # 500/502 — server-side crash
    )
    if litellm is not None
    else ()
)

# How many times to retry after a server-side error survives LiteLLM's own
# built-in num_retries. Delay doubles each attempt (exponential backoff) with
# a small random jitter to avoid a thundering-herd when many workers hit the
# same outage simultaneously.
_SERVER_RETRY_ATTEMPTS = 5
_SERVER_RETRY_BASE_DELAY = 2.0  # seconds


def _coerce_df_index(raw_value):
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


def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_output_csv_path(output_dir, prompt_func_name):
    return os.path.join(output_dir, f"{prompt_func_name}.csv")


def get_processed_df_indices(output_dir, prompt_func_name):
    """
    Read the df_index column from the exact output CSV associated with the
    current task and output location.
    """
    csv_filename = get_output_csv_path(output_dir, prompt_func_name)
    if not os.path.isfile(csv_filename):
        return set()

    processed_indices = set()

    with open(csv_filename, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'df_index' not in reader.fieldnames:
            return set()

        for row in reader:
            df_index = _coerce_df_index(row.get('df_index'))
            if df_index is not None:
                processed_indices.add(df_index)

    return processed_indices


def parse_llm_json(response_text):
    """
    Strips any markdown formatting (like ```json ... ```) and parses the string as JSON.
    Returns the parsed dictionary, or None if parsing fails.
    """
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", response_text, re.DOTALL)
    json_str = match.group(1) if match else response_text.strip()
        
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def append_to_csv(df_index, prompt_func_name, parsed_json, output_dir):
    """
    Appends the parsed JSON to a CSV named after the prompt function.
    Explicitly tracks the ground-truth dataframe index (df_index) as its own column.
    Creates the file and header if it doesn't exist.

    Thread-safe: acquires a per-file lock before touching the filesystem so that
    concurrent workers never interleave their writes or double-write the header.
    """
    ensure_output_dir(output_dir)
    csv_filename = get_output_csv_path(output_dir, prompt_func_name)

    if not isinstance(parsed_json, dict):
        print("Error: Parsed JSON is not a dictionary. Cannot save to CSV.")
        return

    # Ensure df_index is the first column
    fieldnames = ['df_index'] + list(parsed_json.keys())

    # Acquire the lock for this specific CSV before any filesystem interaction.
    # file_exists must be checked inside the lock — without this, two threads
    # could both see file_exists=False and both write a duplicate header row.
    lock = _get_csv_lock(csv_filename)
    with lock:
        file_exists = os.path.isfile(csv_filename)

        with open(csv_filename, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            row_data = {'df_index': df_index}
            row_data.update(parsed_json)

            try:
                writer.writerow(row_data)
            except ValueError as e:
                print(f"Warning: Field name mismatch when writing to CSV: {e}")
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writerow(row_data)

def process_and_store_response(call_llm_func, messages, tools, df_index, prompt_func_name, output_dir):
    """
    Calls the LLM, attempts to parse the JSON. Saves to CSV on success.

    Two-layer retry strategy:

    Outer layer — server-side error retry (_SERVER_RETRY_ATTEMPTS):
        Catches transient infrastructure failures (rate limits, 5xx, timeouts,
        connection errors) that survived LiteLLM's own built-in num_retries.
        Uses exponential backoff with jitter so that many concurrent workers
        don't all hammer the API again at the same moment. On each outer
        attempt, the message list is reset to its original state so we start
        a clean conversation rather than replaying a corrupted one.

    Inner layer — JSON parse retry (max_json_retries):
        Fires when the LLM responds successfully but returns malformed JSON.
        Appends the bad response to the conversation and asks the model to
        correct itself. This is a model-quality issue, not a server issue.
    """
    max_json_retries = 3

    # Snapshot the original messages so the outer server-retry loop can reset
    # to a clean slate if the inner loop polluted the list before the error hit.
    original_messages = list(messages)

    for server_attempt in range(_SERVER_RETRY_ATTEMPTS):
        # Reset to original messages at the start of each server-level attempt
        # so failed JSON retries from a prior attempt don't carry over.
        working_messages = list(original_messages)

        try:
            for json_attempt in range(max_json_retries):
                print(
                    f"Calling LLM for dataframe index {df_index} "
                    f"(server attempt {server_attempt + 1}, "
                    f"JSON attempt {json_attempt + 1})..."
                )
                content, finish_reason, prompt_tokens, completion_tok, total_tokens = (
                    call_llm_func(working_messages, tools)
                )

                parsed_json = parse_llm_json(content)

                if parsed_json is not None:
                    print(f"Successfully parsed JSON for index {df_index}. Saving to CSV.")
                    append_to_csv(df_index, prompt_func_name, parsed_json, output_dir)
                    return parsed_json, content

                print(f"Failed to parse JSON for index {df_index}. Retrying with correction prompt...")

                # Append the malformed response as assistant, then nudge the model
                working_messages.append({"role": "assistant", "content": content})
                working_messages.append({
                    "role": "user",
                    "content": "Your previous response was not valid JSON. Please output ONLY valid JSON without any other text or markdown formatting.",
                })

            # All JSON retries exhausted — the model consistently returned
            # non-JSON. This is a model-quality issue, not a server error,
            # so there is no point in retrying at the server level.
            print(f"Failed to get valid JSON for index {df_index} after {max_json_retries} attempts.")
            return None, None

        except _RETRYABLE_ERRORS as e:
            # A transient server-side error that survived LiteLLM's internal
            # num_retries. Back off and let the server recover before trying again.
            # Jitter (±20%) prevents all workers from waking up simultaneously
            # after a shared outage.
            if server_attempt + 1 == _SERVER_RETRY_ATTEMPTS:
                print(
                    f"Giving up on index {df_index} — server-side error persisted "
                    f"after {_SERVER_RETRY_ATTEMPTS} outer attempts: {e}"
                )
                return None, None

            jitter = random.uniform(0.8, 1.2)
            delay = _SERVER_RETRY_BASE_DELAY * (2 ** server_attempt) * jitter
            print(
                f"Server-side error for index {df_index} "
                f"(attempt {server_attempt + 1}/{_SERVER_RETRY_ATTEMPTS}): {e}. "
                f"Retrying in {delay:.1f}s..."
            )
            time.sleep(delay)

    # Unreachable — the loop always returns or sleeps, but satisfies type checkers.
    return None, None
