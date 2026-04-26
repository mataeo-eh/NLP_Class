import sys
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Prompts.py and LLM_Tools/ both live one level up in NHTSA_Project/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "LLM_Tools"))

from State import State
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from Prompts import (
    Specify_Task_Type,
    Retrieve_Data_Prompt,
    build_schema_summary,
    ANALYSIS_PROMPTS,
)
from config import mercury_llm, gpt5_4_mini_llm, gpt5_1_llm
from LLM_Tools.NHTSA_Query_Tools import (
    get_rows_by_position,
    filter_rows,
    list_csv_files,
    get_csv_schema,
    filter_csv,
    get_csv_rows_by_position,
    Ask_User,
    User_Answer,
)
# Reuse the proven CSV-writing helpers from the original main script. These
# already implement per-file locking (thread-safe), header creation on first
# write, and df_index dedup reads — exactly the behaviour we want to mirror
# from NHTSA_Project_Main.py for the LangGraph CSV-append step.
from Project_Tools.Store_LLM_Responses import (
    append_to_csv,
    ensure_output_dir,
    get_processed_df_indices,
)
# ---------------------------------------------------------------------------
# Load environment variables from a .env file
# ---------------------------------------------------------------------------
load_dotenv()



def classify_task(state: State) -> dict:
    user_request = state["user_request"]

    # Build system + user prompts from Prompts.py
    prompts = Specify_Task_Type(user_request)

    # Construct a messages list so Call_LLM can accept tools and history later
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user",   "content": prompts["user"]},
    ]

    # llm.invoke accepts a list of role/content dicts and returns an AIMessage;
    # .content extracts the text string from that message.
    content = gpt5_4_mini_llm.invoke(messages).content

    # Strip any stray whitespace; the prompt instructs the model to return only the label
    task_type = content.strip()

    return {"task_type": task_type}

def retrieve_data(state: State) -> dict:
    user_request = state["user_request"]
    task_type = state["task_type"]

    # Build the schema string once per call and inject it into the system prompt.
    # This tells the LLM about all 49 parquet columns without requiring a tool call.
    schema_str = build_schema_summary()
    prompts = Retrieve_Data_Prompt(user_request, schema_str)

    # All 8 retrieval tools available in this node.
    # The tool map lets _dispatch_tool look up the callable by name from tool_call dicts.
    tool_list = [
        get_rows_by_position,
        filter_rows,
        list_csv_files,
        get_csv_schema,
        filter_csv,
        get_csv_rows_by_position,
        Ask_User,
        User_Answer,
    ]
    tool_map = {t.name: t for t in tool_list}

    # Tools whose JSON output represents fetched row data we want to preserve as
    # structured query_result for downstream nodes. The other tools — Ask_User,
    # User_Answer, list_csv_files, get_csv_schema — produce metadata or
    # clarification messages, not row data, so their outputs are not captured.
    DATA_TOOLS = {
        "get_rows_by_position",
        "filter_rows",
        "filter_csv",
        "get_csv_rows_by_position",
    }

    llm = gpt5_1_llm.bind_tools(tool_list)

    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user",   "content": prompts["user"]},
    ]

    # Track the most recent successful data-tool result. The LLM may call tools
    # multiple times during the loop; we only keep the final fetched rows so
    # downstream nodes see exactly what the LLM ultimately decided to retrieve.
    last_data_rows: list[dict] = []
    final_response = "Retrieval reached max iterations without a final answer."

    # Bounded agentic loop — allows the LLM to call tools for clarification and data
    # fetching, but prevents runaway execution with a hard iteration cap.
    MAX_ITERATIONS = 8
    for _ in range(MAX_ITERATIONS):
        response = llm.invoke(messages)
        messages.append(response)

        # No tool calls means the LLM has produced its final answer — capture
        # the prose for state["response"] and exit the loop.
        if not response.tool_calls:
            final_response = response.content
            break

        # Execute each tool call and feed the results back into the message history
        for tool_call in response.tool_calls:
            tool_fn = tool_map.get(tool_call["name"])
            if tool_fn is None:
                # Unknown tool — return an error message as the tool result
                result = f"Error: unknown tool '{tool_call['name']}'"
            else:
                result = tool_fn.invoke(tool_call["args"])

            # Capture parsed rows when a data-fetching tool returned a JSON array
            # of row dicts. Non-list payloads (error dicts, "no matches" messages)
            # and parse failures are intentionally ignored: only a real row fetch
            # updates last_data_rows.
            if tool_call["name"] in DATA_TOOLS:
                try:
                    parsed = json.loads(str(result))
                    if isinstance(parsed, list):
                        last_data_rows = parsed
                except (json.JSONDecodeError, TypeError):
                    pass

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    # Shape query_result based on where the data is headed next:
    #   "analyze"  -> project to {index, CDESCR} only. The analyze node's prompts
    #                 are designed to reason over complaint description text, and
    #                 the index lets per-complaint results be written back to a CSV.
    #                 Rows missing CDESCR are dropped — nothing to analyze.
    #   otherwise  -> pass the full row dicts. Pure retrieve tasks exit straight
    #                 to TTS, so the user gets every field the retrieval tool returned.
    if task_type == "analyze":
        query_result = [
            {"index": row["index"], "CDESCR": row["CDESCR"]}
            for row in last_data_rows
            if isinstance(row, dict) and "index" in row and "CDESCR" in row
        ]
    else:
        query_result = last_data_rows

    return {
        "query_result": query_result,
        "response": final_response,
    }

def analyze(state: State) -> dict:
    # Two-step internal pipeline (no inner loop, just sequential dependency):
    #
    #   1. gpt-5.1 reads the user's request and picks ONE analysis prompt by
    #      name from the ANALYSIS_PROMPTS registry. This is the reasoning-heavy
    #      step — choosing the right tool for the job — so it goes to the
    #      stronger model.
    #
    #   2. gpt-5.4-mini runs the chosen prompt once per row in query_result.
    #      Each row carries its original parquet/CSV index plus the CDESCR
    #      complaint text (retrieve_data already projected the dict down to
    #      just those two fields when task_type == "analyze"). Per-row results
    #      preserve the index so downstream consumers can write each analysis
    #      back to a CSV alongside the original complaint.
    user_request = state["user_request"]
    rows = state["query_result"]

    # No rows fetched — nothing to analyze. Return an empty list so downstream
    # consumers (TTS formatter, csv_append) see well-typed results rather than None.
    if not rows:
        return {
            "analysis": [],
            "analysis_prompt_name": "",
            "reasoning_output": "No rows were retrieved; analyze node has nothing to process.",
        }

    # ----- Step 1: prompt selection via gpt-5.1 -----------------------------
    # Build the list of (name, description) pairs dynamically from the registry
    # so adding a new entry to ANALYSIS_PROMPTS in Prompts.py automatically
    # extends the router with no changes here.
    prompt_options = "\n\n".join(
        f"{i+1}. {name}\n   {desc}"
        for i, (name, (_fn, desc)) in enumerate(ANALYSIS_PROMPTS.items())
    )
    valid_names = "\n  ".join(ANALYSIS_PROMPTS.keys())

    selector_system = f'''\
You are a prompt router for an NHTSA vehicle safety complaint analysis pipeline.

Read the user's request and pick exactly ONE analysis prompt to use for processing
each retrieved complaint.

### Available Analysis Prompts

{prompt_options}

### Output Rule

Respond with EXACTLY one of these prompt names and nothing else — no punctuation,
no explanation, no quotes, no newlines, no reasoning:

  {valid_names}

Any response that contains anything other than one of those exact strings is wrong.
'''

    selector_user = f'''\
--- BEGIN USER REQUEST ---
{user_request}
--- END USER REQUEST ---
'''

    selection_raw = gpt5_1_llm.invoke([
        {"role": "system", "content": selector_system},
        {"role": "user",   "content": selector_user},
    ]).content.strip()

    # Validate the model's choice. If the response is malformed (extra text,
    # punctuation, hallucinated prompt name), fall back to the first registry
    # entry so the pipeline still produces output rather than crashing. This
    # is a deterministic fallback, not a guess at intent.
    selected_name = selection_raw if selection_raw in ANALYSIS_PROMPTS else next(iter(ANALYSIS_PROMPTS))
    prompt_fn, _selected_desc = ANALYSIS_PROMPTS[selected_name]

    # ----- Step 2: per-row execution via gpt-5.4-mini -----------------------
    # The chosen prompt function takes a single complaint string and returns
    # {"system": ..., "user": ...}. We call it once per row with that row's
    # CDESCR text, run the lightweight model, and parse the JSON output that
    # every analysis prompt is documented to produce.
    results: list[dict] = []
    for row in rows:
        complaint_text = row.get("CDESCR")
        if not complaint_text:
            # No description text means there is nothing for the prompt to
            # reason over. Skip rather than feeding the model an empty string.
            continue

        prompts = prompt_fn(complaint_text)
        raw = gpt5_4_mini_llm.invoke([
            {"role": "system", "content": prompts["system"]},
            {"role": "user",   "content": prompts["user"]},
        ]).content.strip()

        # All analysis prompts are explicitly documented to return JSON only.
        # If the model ignores that instruction, store the raw text under an
        # "error" wrapper so the row's index is still preserved and the
        # downstream CSV writer can decide what to do with it.
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"error": "Model returned non-JSON output", "raw": raw}

        results.append({"index": row["index"], "result": parsed})

    return {
        "analysis": results,
        # Pass the selected prompt name through state explicitly. csv_append
        # consumes this to decide which CSV file to write to (one CSV per
        # prompt, matching the convention of NHTSA_Project_Main.py).
        "analysis_prompt_name": selected_name,
        # Surface which prompt won the routing step. This is exposed via TTS
        # (reasoning_output) so the user hears the system explain its choice.
        "reasoning_output": f"Selected analysis prompt: {selected_name}",
    }

# ---------------------------------------------------------------------------
# CSV append output directory
# ---------------------------------------------------------------------------
# Resolve the Outputs directory once at module load, the same way NHTSA_Project_Main.py
# resolves it relative to the project root. From LangGraph/Nodes.py the path is:
#   <repo>/Unit2/NHTSA_Project/LangGraph/Nodes.py  ->  parent.parent  ->  NHTSA_Project/
# This matches the path NHTSA_Query_Tools.OUTPUTS_DIR uses, keeping all CSV
# producers and consumers writing/reading from one canonical location.
OUTPUTS_DIR = str(Path(__file__).resolve().parent.parent / "Outputs")


def csv_append(state: State) -> dict:
    """
    Append per-row analysis results to the CSV that matches the chosen prompt.

    Mirrors the CSV-appending behaviour of NHTSA_Project_Main.py:
      - One CSV per prompt (file name = prompt function name + ".csv")
      - df_index is the first column and uniquely identifies each appended row
      - Concurrent writes are safe because append_to_csv uses a per-file lock
      - Rows whose df_index already exists in the CSV are skipped, so re-running
        analysis on a complaint never produces duplicate entries

    No LLM calls happen here. The analyze node already produced parsed JSON
    results; this node just routes them to disk in parallel.
    """
    analysis = state["analysis"]
    prompt_name = state.get("analysis_prompt_name", "")

    # Empty analysis or missing prompt name means there is nothing to write.
    # Both can happen on the "no rows retrieved" branch of analyze, or on a
    # task_type path where analyze never ran.
    if not analysis or not prompt_name:
        return {}

    ensure_output_dir(OUTPUTS_DIR)

    # Read the existing df_index values once, before launching workers. Doing
    # this in parallel per worker would be wasteful (every thread would re-read
    # the same CSV) and racy (one thread's append could be visible to another's
    # read mid-batch, producing inconsistent skip decisions). A single up-front
    # snapshot is the deterministic equivalent of the main script's behaviour.
    already_processed = get_processed_df_indices(OUTPUTS_DIR, prompt_name)

    # Filter to entries that need writing:
    #   1. Skip indices already in the CSV — the dedup guarantee the user requested.
    #   2. Skip entries whose result is not a clean dict, or has the "error" key
    #      that analyze attaches when JSON parsing failed. Writing those would
    #      pollute the CSV header (the first row written defines the column set,
    #      and an error wrapper has different keys than a real analysis dict).
    pending = []
    for entry in analysis:
        idx = entry.get("index")
        result = entry.get("result")
        if idx is None or idx in already_processed:
            continue
        if not isinstance(result, dict) or "error" in result:
            continue
        pending.append(entry)

    if not pending:
        print(f"[csv_append] Nothing to write for {prompt_name} "
              f"(all {len(analysis)} indices already present or invalid).")
        return {}

    # Per-row writer. Returns (index, success_bool) so the main thread can
    # tally results without sharing mutable state across workers.
    def write_one(entry: dict) -> tuple[int, bool]:
        try:
            append_to_csv(
                df_index=entry["index"],
                prompt_func_name=prompt_name,
                parsed_json=entry["result"],
                output_dir=OUTPUTS_DIR,
            )
            return entry["index"], True
        except Exception as exc:
            # Catch broadly so one bad entry never aborts the whole batch.
            # The original main script does the same — it logs and continues.
            print(f"[csv_append] Failed to append index {entry['index']}: {exc}")
            return entry["index"], False

    # Parallel append. append_to_csv is internally thread-safe via per-file
    # locks, so workers writing to the SAME CSV serialize only for the
    # microseconds of the actual write. With max_workers=None the executor
    # defaults to min(32, os.cpu_count() + 4), which mirrors the original
    # script's "all available system workers" default behaviour.
    written = 0
    failed = 0
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(write_one, entry): entry["index"] for entry in pending}
        for future in as_completed(futures):
            _idx, ok = future.result()
            if ok:
                written += 1
            else:
                failed += 1

    print(f"[csv_append] {prompt_name}: wrote {written} new row(s), "
          f"skipped {len(analysis) - len(pending)} duplicate/invalid, "
          f"{failed} failure(s).")

    return {}


def agentic_loop(state: State) -> dict:
    user_request = state["user_request"]

    # imagine LLM call here that reads the request
    # Puts the result of the query, such as a content retrieval, and adds it to the state
    query_result = "query_result"  # placeholder for now

    return {"query_result": query_result}


def fetch_from_database(state: State) -> dict:
    user_request = state["user_request"]
    
    # imagine LLM call here that reads the request
    # Puts the result of the query, such as a content retrieval, and adds it to the state
    query_result = "query_result"  # placeholder for now
    
    return {"query_result": query_result}