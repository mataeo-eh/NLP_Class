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
    Specify_Agentic_Subtype,   # sub-classifier prompt; added by W5
    Retrieve_Data_Prompt,
    Agentic_Analyze_Prompt,    # system+user prompts for the agentic_analyze node; added by W6
    Agentic_Explore_Prompt,    # system+user prompts for the agentic_explore node; added by W7
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
# voice_ask_user is a LangChain @tool that speaks a question aloud via TTS and
# returns the user's spoken reply as a string. Used by agentic_analyze to allow
# the LLM to request clarification mid-loop when the user's intent is ambiguous.
from Project_Tools.Voice_Tools import voice_ask_user
from Project_Tools.Audio_Playback import narrate_node_result
# Chart tools used by agentic_explore: render bar charts, model-year charts, and
# LLM-vs-NHTSA comparison charts; describe previously-rendered charts by name;
# cleanup_temp_charts deletes temp PNGs (non-macOS) after the loop exits.
from Project_Tools.Chart_Tools import (
    create_bar_chart_tool,
    create_model_year_chart_tool,
    create_human_subsystem_frequency_chart_tool,
    compare_LLM_to_NHTSA_tool,
    describe_chart_data,
    cleanup_temp_charts,
)
# code_exec is a LangChain @tool that executes a Python snippet ONLY after the
# user explicitly approves it via voice — each call requires fresh permission.
from Project_Tools.Code_Exec_Tools import code_exec
# Codebase inspection tools: let the model list and read sections of project
# source files so it can answer questions about pipeline internals.
from Project_Tools.Codebase_Tools import list_project_files, read_file_section
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

    narrate_node_result("classify_task", {"task_type": task_type})

    return {"task_type": task_type}


def classify_agentic_subtype(state: State) -> dict:
    """
    SUB-CLASSIFIER — runs only after classify_task has already confirmed that
    task_type == "agentic_retrieve_and_analyze".  It reads the user's original
    request and asks the LLM (via Specify_Agentic_Subtype from Prompts.py) to
    pick one of exactly two sub-types:

      "agentic_analyze"
          The user has a concrete, answerable question.  The agent should
          retrieve targeted rows from the NHTSA database and produce a
          structured analysis output (safety categories, subsystems, etc.)
          — essentially the agentic version of the non-agentic "analyze" path.

      "agentic_explore"
          The user wants open-ended discovery: browse broadly, surface
          patterns, compare across complaints, and narrate findings without
          a predetermined output schema.  The agent decides on its own when
          it has collected enough evidence to summarise.

    Allowed outputs: exactly "agentic_analyze" or "agentic_explore".
    Any other string will be caught by route_agentic_subtype (Edges.py) and
    gracefully defaulted to "agentic_explore".

    Mirrors the structure of classify_task — same message-building pattern,
    same gpt5_4_mini_llm call, same .content.strip() extraction.

    Returns:
        dict with key "agentic_subtype" set to the stripped LLM output.
        LangGraph merges this dict into the running State automatically.
    """
    user_request = state["user_request"]

    # Build system + user prompts from Prompts.py (W5 adds Specify_Agentic_Subtype)
    prompts = Specify_Agentic_Subtype(user_request)

    # Construct a messages list in the same role/content format used everywhere
    # else in this file — this keeps LLM invocation consistent and makes it
    # straightforward to add history or tool context in the future.
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user",   "content": prompts["user"]},
    ]

    # gpt5_4_mini_llm is the fast, cheap model — appropriate for a binary
    # classification task where we only need a single label word, not deep
    # reasoning.  .invoke returns an AIMessage; .content gives the text string.
    content = gpt5_4_mini_llm.invoke(messages).content

    # Strip any stray whitespace; the prompt instructs the model to return only
    # the label ("agentic_analyze" or "agentic_explore").
    agentic_subtype = content.strip()

    narrate_node_result("classify_agentic_subtype", {"agentic_subtype": agentic_subtype})

    return {"agentic_subtype": agentic_subtype}


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

    narrate_node_result("retrieve_data", {
        "row_count": len(query_result),
        "task_type": state.get("task_type", ""),
    })

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

    narrate_node_result("analyze_prompt_selected", {
        "analysis_prompt_name": selected_name,
    })

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

    narrate_node_result("analyze_complete", {
        "row_count": len(results),
        "analysis_prompt_name": selected_name,
    })

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


def confirm_csv_write(state: State) -> dict:
    """
    User-confirmation gate that sits between the analyze node and csv_append.

    Runtime contract
    ----------------
      - Reads:  state["analysis"], state["analysis_prompt_name"]
      - Writes: state["csv_write_confirmed"] (bool)
      - Side effects: one Mercury+TTS narration call (via narrate_node_result)
        plus one voice_ask_user call (TTS prompt + STT capture in audio mode,
        or print + input() in text mode).

    Why this node exists
    --------------------
    Originally analyze routed straight into csv_append, which meant the pipeline
    always wrote analysis results to disk. That's correct for local interactive
    runs, but breaks the moment the pipeline is hosted somewhere the agent has
    no filesystem write permission. Inserting an explicit yes/no gate means the
    same graph runs safely in both deployment modes — the user is the source of
    truth for "should we touch disk?" and answering "no" simply skips csv_append.

    Behaviour
    ---------
      1. If there are no analysis rows, or no prompt name was selected, there is
         nothing to commit. Short-circuit to "not confirmed" and skip the spoken
         question entirely — asking "save these zero results?" would be a poor UX.
      2. Use narrate_node_result (Mercury + TTS) to summarise what is queued for
         write. This is the same inter-chain reasoning narration channel every
         other node uses, so the user hears one consistent voice across the run.
      3. Ask the user a literal yes/no question via voice_ask_user. That tool
         already handles audio-vs-text mode internally — in audio mode it speaks
         the question and returns a Whisper transcript; in text mode it prints
         and reads stdin — so this node does not need to branch on audio mode.
      4. Parse the reply with simple keyword matching against canonical yes/no
         tokens. The user has been instructed to answer "yes" or "no", and the
         project explicitly tolerates the user being trusted to comply. Any
         reply that matches neither token set is treated as "no" because not
         writing is the safer default.
    """
    analysis = state["analysis"]
    prompt_name = state.get("analysis_prompt_name", "")

    # Nothing to commit -> skip the prompt entirely. csv_append already no-ops
    # in this case, but we'd rather not subject the user to a spoken question
    # about a write that wouldn't happen anyway.
    if not analysis or not prompt_name:
        print("[confirm_csv_write] Nothing to write; skipping confirmation prompt.")
        return {"csv_write_confirmed": False}

    # Mercury+TTS narration of the pending write. narrate_node_result feeds
    # the context dict into Node_Progress_Summary_Prompt, which Mercury formats
    # into 2-4 sentences of TTS-safe prose. This gives the user spoken context
    # about what is queued before they have to make the yes/no decision.
    narrate_node_result("confirm_csv_write", {
        "analysis_prompt_name": prompt_name,
        "row_count": len(analysis),
        "destination_csv": f"Outputs/{prompt_name}.csv",
        "next_step_if_yes": "write rows to CSV via csv_append",
        "next_step_if_no":  "skip the write and end the run",
    })

    # Explicit yes/no question. voice_ask_user.invoke is the LangChain @tool
    # contract — same one used by the agentic nodes — so audio-mode TTS+STT
    # vs text-mode stdin handling is encapsulated inside the tool.
    question = (
        f"I have {len(analysis)} analysis result"
        f"{'s' if len(analysis) != 1 else ''} ready to write to the "
        f"{prompt_name} CSV file. Would you like me to save them? "
        "Please answer yes or no."
    )
    raw_reply = voice_ask_user.invoke({"question": question})

    # Lightweight token-matching yes/no parser. The user was asked to answer
    # literally "yes" or "no", but we accept a short list of common natural
    # variants on each side so the dialogue feels less rigid. Order matters:
    # checking yes_tokens first means a hypothetical reply like "yes, don't
    # bother" would be treated as a yes, but the user has been instructed to
    # answer cleanly and we tolerate that assumption per the project decision.
    reply_lower = raw_reply.strip().lower() if isinstance(raw_reply, str) else ""
    yes_tokens = ("yes", "yeah", "yep", "yup", "sure", "affirmative",
                  "go ahead", "do it", "save them", "save it", "please")
    no_tokens  = ("no", "nope", "nah", "negative", "don't", "do not",
                  "skip", "cancel", "stop")

    if any(tok in reply_lower for tok in yes_tokens):
        confirmed = True
    elif any(tok in reply_lower for tok in no_tokens):
        confirmed = False
    else:
        # Ambiguous or empty reply -> default to NOT writing. The user can
        # re-run the pipeline if they actually wanted the write to happen;
        # there's no equivalent way to undo an unwanted CSV mutation.
        print(f"[confirm_csv_write] Ambiguous reply {raw_reply!r}; "
              f"defaulting to skip CSV write.")
        confirmed = False

    # Narrate the decision so the user hears confirmation that the system
    # understood their answer, before the graph routes to csv_append or END.
    narrate_node_result("confirm_csv_write_decision", {
        "decision": "writing to CSV" if confirmed else "skipping CSV write",
        "analysis_prompt_name": prompt_name,
        "row_count": len(analysis),
    })

    return {"csv_write_confirmed": confirmed}


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


def agentic_analyze(state: State) -> dict:
    # PURPOSE: Row-level reasoning agent for the "agentic" task branch.
    # Unlike retrieve_data (which fetches rows for a downstream analyze node) or
    # analyze (which runs a fixed prompt per row), agentic_analyze is a single
    # self-contained loop: it retrieves NHTSA complaint rows, optionally pulls
    # supporting columns, and emits a final structured JSON verdict in one pass.
    # The verdict is not written to CSV — it is ephemeral, returned via state
    # ["response"] and delivered to the user via TTS by the graph's main runner.
    #
    # MODEL: gpt5_1_llm (the strongest reasoning + tool orchestration model in
    # this pipeline). Same model used by the retrieve_data node's LLM binding.
    #
    # TOOLS: All six NHTSA query tools (same set as retrieve_data) plus
    # voice_ask_user for mid-loop clarification. Chart tools and code execution
    # tools are intentionally excluded — this node's output is spoken, not visual.
    #
    # PARALLEL DISPATCH: When the model emits multiple tool_calls in one turn,
    # all are submitted concurrently via ThreadPoolExecutor. This mirrors the
    # parallel write strategy in csv_append and reduces round-trips when the model
    # batches independent fetches (e.g., several row-range retrievals at once).
    # Results are collected via as_completed and appended as ToolMessages in
    # completion order before the next LLM turn.
    #
    # MAX_ITERATIONS = 12 (vs retrieve_data's 8). Exploration + verdict
    # composition typically requires more tool-call/response cycles than pure
    # retrieval, so the cap is raised to give the model enough turns.
    #
    # RETURNS: dict with:
    #   "response"        — final str (the structured JSON verdict or fallback msg)
    #   "iteration_log"   — list[dict] appended per iteration (carries forward any
    #                       prior entries from state["iteration_log"])
    #   "agentic_subtype" — hardcoded "agentic_analyze" so downstream routing
    #                       and W7 nodes can distinguish sub-branches
    # Does NOT touch: analysis, analysis_prompt_name, query_result fields —
    # those belong to the analyze/retrieve_data branch, not this node.

    user_request = state["user_request"]
    schema_str   = build_schema_summary()

    # Build the prompt pair (system context + user request). Agentic_Analyze_Prompt
    # returns {"system": <str>, "user": <str>}. We adopt the same dict-based message
    # format that retrieve_data uses so the pattern is consistent across all nodes.
    prompts = Agentic_Analyze_Prompt(user_request, schema_str)
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user",   "content": prompts["user"]},
    ]

    # All tools available to this node. voice_ask_user enables mid-loop TTS
    # clarification. No chart or code-execution tools — output is spoken JSON.
    tool_list = [
        get_rows_by_position,
        filter_rows,
        list_csv_files,
        get_csv_schema,
        filter_csv,
        get_csv_rows_by_position,
        voice_ask_user,
    ]
    # Dict keyed by tool name for O(1) dispatch inside the loop.
    tool_map = {t.name: t for t in tool_list}

    # Bind tools to the model. This adds the tool schema to the model's context
    # so it knows how to structure tool_call dicts in its responses.
    llm = gpt5_1_llm.bind_tools(tool_list)

    # Higher cap than retrieve_data (8) because exploration + verdict composition
    # needs more turns than pure row retrieval.
    MAX_ITERATIONS = 12

    # Carry forward any log entries produced by earlier nodes in the same run
    # (e.g., classify_agentic_subtype might append a classification entry).
    iteration_log = list(state.get("iteration_log") or [])
    final_response = ""

    for i in range(MAX_ITERATIONS):
        response = llm.invoke(messages)
        messages.append(response)

        # No tool calls → the model has produced its final answer. Capture the
        # content string and break out of the loop cleanly.
        if not getattr(response, "tool_calls", None):
            final_response = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            iteration_log.append({
                "iter":       i,
                "model":      "gpt-5.1",
                "tool_calls": [],
                "summary":    "final response emitted (no tool calls)",
            })
            break

        tool_calls = response.tool_calls  # list of dicts: {name, args, id}

        # --- Parallel dispatch ---------------------------------------------------
        # Submit all tool calls concurrently. Independent fetches (e.g., several
        # row-range lookups, or a schema check alongside a filter call) complete
        # in parallel rather than sequentially, reducing total wall time.
        # Unknown tools receive an immediate error ToolMessage without spawning a
        # future so the model knows what went wrong on the next turn.
        with ThreadPoolExecutor(max_workers=max(1, len(tool_calls))) as executor:
            futures = {}
            for tc in tool_calls:
                tool = tool_map.get(tc["name"])
                if tool is None:
                    # Unknown tool — return an error immediately without a future.
                    # Appending now (before as_completed) is safe because the
                    # executor hasn't yielded this tc in any future.
                    messages.append(ToolMessage(
                        content=f"Tool error: unknown tool {tc['name']!r}",
                        tool_call_id=tc["id"],
                    ))
                    continue
                futures[executor.submit(tool.invoke, tc["args"])] = tc

            # Collect results as they complete (order not guaranteed, which is fine —
            # each ToolMessage carries its tool_call_id for the model to correlate).
            for fut in as_completed(futures):
                tc = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    # Surface tool errors as ToolMessages so the model can adapt
                    # rather than silently losing a result.
                    result = f"Tool error: {exc}"
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))
        # ------------------------------------------------------------------------

        iteration_log.append({
            "iter":       i,
            "model":      "gpt-5.1",
            "tool_calls": [tc["name"] for tc in tool_calls],
            "summary":    f"dispatched {len(tool_calls)} tool call(s) in parallel",
        })

        narrate_node_result("agentic_iteration", {
            "iter":       i,
            "max_iter":   MAX_ITERATIONS,
            "tool_calls": [tc["name"] for tc in tool_calls],
            "reasoning":  f"dispatched {len(tool_calls)} tool call(s) in parallel",
        })

    else:
        # Loop exhausted all 12 iterations without the model producing a tool-free
        # response. Record the cap-fallthrough and return a graceful degradation msg.
        final_response = "Agentic analyze reached max iterations without a final answer."
        iteration_log.append({
            "iter":       MAX_ITERATIONS,
            "model":      "gpt-5.1",
            "tool_calls": [],
            "summary":    "iteration cap fallthrough",
        })

    return {
        "response":        final_response,
        "iteration_log":   iteration_log,
        "agentic_subtype": "agentic_analyze",
    }


def agentic_explore(state: State) -> dict:
    # PURPOSE: Conversational exploration agent for the "agentic" task branch.
    # While agentic_analyze is a one-shot verdict emitter (retrieves rows, judges,
    # returns structured JSON), agentic_explore is a multi-turn dialogue: the user
    # can ask follow-up questions, request charts, run ad-hoc code, and browse the
    # NHTSA complaints data interactively. The final output is a short spoken
    # summary (TTS), not a structured JSON verdict.
    #
    # VOICE-EVERYWHERE UX: voice_ask_user is always in the tool list so the model
    # can proactively check in with the user rather than guessing at intent. This
    # is especially important when chart results need narration (the user cannot
    # see PNG bytes) or when code_exec is about to be called (explicit permission
    # is required every time).
    #
    # PARALLEL DISPATCH: When the model emits multiple tool_calls in one turn,
    # all are submitted concurrently via ThreadPoolExecutor. This mirrors the
    # parallel dispatch strategy in agentic_analyze and reduces round-trips when
    # the model batches independent operations (e.g., a filter call + a schema
    # lookup, or two independent chart renders).
    #
    # CODE_EXEC PERMISSION GATE: code_exec requires the user to approve every
    # individual execution via voice. If the user denies, the model must NOT
    # retry the same code; it should pivot to a different approach or ask the
    # user for guidance via voice_ask_user. This gate is enforced inside
    # code_exec itself — this node does not add a second layer.
    #
    # MAX_ITERATIONS = 12: Same cap as agentic_analyze. Conversational exploration
    # can be long (chart request -> narration -> follow-up -> another chart), so
    # 12 turns gives enough room without risking unbounded loops.
    #
    # MODEL: gpt5_4_mini_llm (the faster "mini" model). agentic_explore is
    # conversational and benefits from lower latency — the user is waiting on TTS
    # responses in real time. By contrast, agentic_analyze uses gpt5_1_llm because
    # its structured JSON verdict demands the strongest reasoning. Here, speed
    # and dialogue fluency are more important than maximum reasoning power.
    #
    # CLEANUP: After the loop ends, cleanup_temp_charts() deletes any temp PNG
    # files that were written to disk on non-macOS platforms during chart calls.
    # macOS Preview-stdin path produces no temp files, so this is a no-op there.
    # Failure to clean up must not propagate — we wrap in try/except.
    #
    # RETURNS: dict with:
    #   "response"        — final str (short spoken summary or fallback msg)
    #   "iteration_log"   — list[dict] appended per iteration (carries forward any
    #                       prior entries from state["iteration_log"])
    #   "agentic_subtype" — hardcoded "agentic_explore" so downstream routing
    #                       can distinguish from agentic_analyze sub-branch
    # Does NOT touch: analysis, analysis_prompt_name, query_result fields —
    # those belong to the analyze/retrieve_data branch, not this node.

    user_request = state["user_request"]
    schema_str   = build_schema_summary()

    # Build the prompt pair (system context + user request). Agentic_Explore_Prompt
    # returns {"system": <str>, "user": <str>}. We use the same dict-based message
    # format as agentic_analyze for consistency across all nodes in the pipeline.
    prompts = Agentic_Explore_Prompt(user_request, schema_str)
    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user",   "content": prompts["user"]},
    ]

    # Full tool list for explore: chart rendering + narration, code execution
    # (with user permission gate), codebase inspection, voice dialogue, and all
    # six NHTSA query tools for data access. 14 tools total.
    tool_list = [
        create_bar_chart_tool,
        create_model_year_chart_tool,
        create_human_subsystem_frequency_chart_tool,
        compare_LLM_to_NHTSA_tool,
        describe_chart_data,
        code_exec,
        list_project_files,
        read_file_section,
        voice_ask_user,
        get_rows_by_position,
        filter_rows,
        list_csv_files,
        get_csv_schema,
        filter_csv,
        get_csv_rows_by_position,
    ]
    # Dict keyed by tool name for O(1) dispatch inside the loop.
    tool_map = {t.name: t for t in tool_list}

    # Bind tools to the mini model. Lower latency than gpt5_1_llm matters here
    # because the user is in a real-time spoken dialogue and perceives each
    # model turn as a pause before TTS plays.
    llm = gpt5_4_mini_llm.bind_tools(tool_list)

    # Higher cap than retrieve_data (8) because conversational exploration
    # can require many chart + narration + follow-up cycles.
    MAX_ITERATIONS = 12

    # Carry forward any log entries produced by earlier nodes in the same run
    # (e.g., classify_agentic_subtype might append a classification entry).
    iteration_log = list(state.get("iteration_log") or [])
    final_response = ""

    for i in range(MAX_ITERATIONS):
        response = llm.invoke(messages)
        messages.append(response)

        # No tool calls -> the model has produced its final conversational answer.
        # Capture the content string and break out of the loop cleanly.
        if not getattr(response, "tool_calls", None):
            final_response = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            iteration_log.append({
                "iter":       i,
                "model":      "gpt-5.4-mini",
                "tool_calls": [],
                "summary":    "final response emitted (no tool calls)",
            })
            break

        tool_calls = response.tool_calls  # list of dicts: {name, args, id}

        # --- Parallel dispatch ---------------------------------------------------
        # Submit all tool calls concurrently. Independent operations (e.g., a chart
        # render + a schema lookup, or two separate filter calls) complete in
        # parallel rather than sequentially, reducing total wall time during the
        # interactive dialogue loop.
        # Unknown tools receive an immediate error ToolMessage without spawning a
        # future so the model knows what went wrong on the next turn.
        with ThreadPoolExecutor(max_workers=max(1, len(tool_calls))) as executor:
            futures = {}
            for tc in tool_calls:
                tool = tool_map.get(tc["name"])
                if tool is None:
                    # Unknown tool — return an error immediately without a future.
                    # Appending now (before as_completed) is safe because the
                    # executor hasn't yielded this tc in any future.
                    messages.append(ToolMessage(
                        content=f"Tool error: unknown tool {tc['name']!r}",
                        tool_call_id=tc["id"],
                    ))
                    continue
                futures[executor.submit(tool.invoke, tc["args"])] = tc

            # Collect results as they complete (order not guaranteed, which is fine —
            # each ToolMessage carries its tool_call_id for the model to correlate).
            for fut in as_completed(futures):
                tc = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    # Surface tool errors as ToolMessages so the model can adapt
                    # rather than silently losing a result.
                    result = f"Tool error: {exc}"
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                ))
        # ------------------------------------------------------------------------

        iteration_log.append({
            "iter":       i,
            "model":      "gpt-5.4-mini",
            "tool_calls": [tc["name"] for tc in tool_calls],
            "summary":    f"dispatched {len(tool_calls)} tool call(s) in parallel",
        })

        narrate_node_result("agentic_iteration", {
            "iter":       i,
            "max_iter":   MAX_ITERATIONS,
            "tool_calls": [tc["name"] for tc in tool_calls],
            "reasoning":  f"dispatched {len(tool_calls)} tool call(s) in parallel",
        })

    else:
        # Loop exhausted all 12 iterations without the model producing a tool-free
        # response. Record the cap-fallthrough and return a graceful degradation msg.
        final_response = "Agentic explore reached max iterations without a final answer."
        iteration_log.append({
            "iter":       MAX_ITERATIONS,
            "model":      "gpt-5.4-mini",
            "tool_calls": [],
            "summary":    "iteration cap fallthrough",
        })

    # --- Temp chart cleanup -----------------------------------------------------
    # Why: temp PNGs are kept around during the loop so the user can re-open them
    # if needed; we clean up after the loop ends. macOS Preview-stdin path produces
    # no temp files so this is a no-op there.
    try:
        cleanup_temp_charts()
    except Exception:
        # Cleanup failure must not propagate — the loop has already completed and
        # the response is ready. Silently swallow so the caller always gets a result.
        pass
    # ----------------------------------------------------------------------------

    return {
        "response":        final_response,
        "iteration_log":   iteration_log,
        "agentic_subtype": "agentic_explore",
    }
