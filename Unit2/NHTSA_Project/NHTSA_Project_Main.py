import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import os
import litellm
import argparse
import re
from litellm import completion
from dotenv import load_dotenv

from Prompts import Safety_Prompt, get_available_prompts, get_prompt_display_name
from Project_Tools.Random_Sampling import (
    get_random_sample_rows,
    get_sequential_rows,
)




# ---------------------------------------------------------------------------
# Load environment variables from a .env file
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# Environment variable guards
# ---------------------------------------------------------------------------
try:
    api_key = os.environ["LITELLM_API_KEY"]
except KeyError:
    raise EnvironmentError(
        "LITELLM_API_KEY is not set. "
        "Export your provider API key before running this script, e.g.:\n"
        "  export LITELLM_API_KEY='sk-...'"
    )

try:
    api_base = os.environ["LITELLM_API_BASE"]
except KeyError:
    raise EnvironmentError(
        "LITELLM_API_BASE is not set. "
        "Export the base URL of your LiteLLM proxy or provider endpoint, e.g.:\n"
        "  export LITELLM_API_BASE='http://localhost:4000'"
    )

try:
    model = os.environ["LITELLM_MODEL"]
except KeyError:
    raise EnvironmentError(
        "LITELLM_MODEL is not set. "
        "Export the model identifier in 'provider/model-name' format, e.g.:\n"
        "  export LITELLM_MODEL='openai/gpt-4o'"
    )

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
# We load from a pre-built Parquet file rather than the raw NHTSA flat file.
#
# Why a cleaned Parquet instead of the raw tab-delimited text file
# ----------------------------------------------------------------
# NHTSA's raw data model represents a single consumer complaint as *multiple
# rows* whenever that complaint involves more than one component subsystem —
# one row per COMPDESC value, with the complaint text (CDESCR) copy-pasted
# identically across all of them. This means a complaint flagged under both
# "ELECTRICAL SYSTEM" and "VEHICLE SPEED CONTROL" appears as two rows that
# are byte-for-byte identical except for CMPLID and COMPDESC.
#
# Build_DF.py (NHTSA/Build_DF.py) collapses these duplicate rows into a
# single row per complaint, where COMPDESC becomes a list of all subsystem
# labels NHTSA associated with it. Grouping is done on ODINO — NHTSA's own
# internal reference number that is shared across all rows belonging to the
# same complaint. Any other columns that differ within a group (e.g., CMPLID,
# or MAKETXT/VIN in multi-party complaints involving a vehicle and attached
# equipment) also become lists.
#
# The result is a 1-complaint-per-row representation that avoids inflating
# complaint counts, double-counting incidents in aggregate statistics, and
# sending the same complaint text to the LLM multiple times.
#
# To regenerate the Parquet after a data refresh:
#   python3 NHTSA/Build_DF.py
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
file_path = PROJECT_ROOT / "NHTSA" / "complaints_cleaned.parquet"


def load_df():
    # Parquet preserves column types from the build step — no manual
    # type conversion needed here. List-typed columns (e.g., COMPDESC)
    # come back as numpy arrays, which behave like lists for iteration.
    return pd.read_parquet(file_path)


def LLM_Message(system_prompt: str = "You are a helpful assistant.", user_prompt: str = "Explain large language models in one paragraph."):
    messages = [
    {
        "role": "system",
        "content": system_prompt,
        # "name": "optional_author_name"   # Max 64 chars. Required when role="function".
    },
    {
        "role": "user",
        "content": user_prompt,
    },
    ]
    return messages

def LLM_Tools():
    tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            # description: Tells the model what the function does so it can
            # decide when to invoke it.
            "description": "Get the current weather for a given city.",
            # parameters: JSON Schema describing the expected arguments.
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Boston'",
                    }
                },
                "required": ["city"],
            },
        },
    }
    ]
    return tools

def Call_LLM(messages = None, tools = None, on_chunk = None):
    response = completion(

    model=model,
    # Which model to call. LiteLLM uses a "provider/model-name" format,
    # e.g. "openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022",
    # "ollama/llama3". The provider prefix tells LiteLLM which SDK adapter
    # to use under the hood.

    messages=messages,
    # The conversation history. See the `messages` list above.

    api_key=api_key,
    # The API key sent to the upstream provider. Overrides any key already
    # set via environment variable for this single call. Useful when rotating keys or calling multiple providers in the same script.

    api_base=api_base,
    # The base URL of the endpoint to hit. Required when pointing at a

    temperature=0.2,
    # Controls randomness. Range: 0.0 – 2.0.
    
    max_tokens=8000,
    # Maximum number of tokens the model may generate in its response.

    tools=tools,
    # List of tool/function schemas the model may call. 

    tool_choice="auto",
    # Controls whether and which tool(s) the model uses.

    stream=True,

    timeout=60,
    # Seconds before LiteLLM raises an APITimeoutError if the provider
    # has not responded. Prevents the script from hanging indefinitely.
    # Catch with: except openai.APITimeoutError.

    num_retries=5,
    # LiteLLM will automatically retry this call up to 3 times on transient
    # server-side errors (rate limits, 5xx, timeouts) before raising. This is
    # the first line of defense; process_and_store_response adds an outer retry
    # layer on top for errors that exhaust even these attempts.

        # ------------------------------------------------------------------
    # DEBUGGING (LiteLLM-specific)
    # ------------------------------------------------------------------

    drop_params=True,
    # When True, LiteLLM silently drops any parameter that the target
    # provider does not support, instead of raising an error. Useful when
    # writing provider-agnostic code that passes all params regardless of
    # the backend. Commented out here so unsupported params surface as errors during development.

    stream_options={"include_usage": True},
    )
    content        = ""
    finish_reason  = None
    prompt_tokens  = 0
    completion_tok = 0
    total_tokens   = 0

    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
            if on_chunk:
                on_chunk(delta.content)
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        if hasattr(chunk, "usage") and chunk.usage:
            prompt_tokens  = chunk.usage.prompt_tokens or 0
            completion_tok = chunk.usage.completion_tokens or 0
            total_tokens   = chunk.usage.total_tokens or 0

    return content, finish_reason, prompt_tokens, completion_tok, total_tokens


def print_LLM_response(content, finish_reason, prompt_tokens, completion_tok, total_tokens):
    print("=== Response ===")
    print(content)
    print()
    print(f"Finish reason : {finish_reason}")
    print(f"Prompt tokens : {prompt_tokens}")
    print(f"Completion tok: {completion_tok}")
    print(f"Total tokens  : {total_tokens}")

from Project_Tools.Store_LLM_Responses import (
    get_processed_df_indices,
    process_and_store_response,
)


def resolve_num_samples(samples_arg, random_sampling=False):
    if samples_arg is None:
        return 10

    normalized_value = samples_arg.strip().lower()

    if normalized_value in ["all", "none"]:
        if random_sampling:
            raise ValueError("--random-sampling requires --samples to be an integer.")
        return None

    try:
        num_samples = int(samples_arg)
    except ValueError:
        raise ValueError(
            f"Invalid --samples value: '{samples_arg}'. "
            "Use a non-negative integer, 'All', or 'None'."
        )

    if num_samples < 0:
        raise ValueError("--samples must be a non-negative integer.")

    return num_samples


def get_complaint_column(df):
    for column_name in ["CDESCR", "complaint", "Complaint", "COMPLAINT"]:
        if column_name in df.columns:
            return column_name

    if len(df.columns) > 19:
        return df.columns[19]

    raise KeyError("Could not determine the complaint text column in the dataset.")


def normalize_prompt_selector(value):
    return re.sub(r"[^a-z0-9]", "", value.lower())


def parse_prompt_selection(selection_text, available_prompts):
    if not selection_text:
        raise ValueError("Prompt selection was not provided.")

    raw_selectors = [value.strip() for value in selection_text.split(",")]
    if any(not value for value in raw_selectors):
        raise ValueError("Invalid prompt selection. Use comma-separated prompt numbers or names.")

    prompt_lookup = {}
    for idx, prompt_func in enumerate(available_prompts, start=1):
        display_name = get_prompt_display_name(prompt_func)
        selector_values = {
            str(idx),
            normalize_prompt_selector(prompt_func.__name__),
            normalize_prompt_selector(display_name),
        }

        if prompt_func.__name__.endswith("_Prompt"):
            selector_values.add(normalize_prompt_selector(prompt_func.__name__[:-7]))

        for selector_value in selector_values:
            prompt_lookup[selector_value] = prompt_func

    selected_prompt_functions = []
    seen_prompt_names = set()

    for raw_selector in raw_selectors:
        selector_key = normalize_prompt_selector(raw_selector)
        prompt_func = prompt_lookup.get(selector_key)
        if prompt_func is None:
            raise ValueError(
                f"Invalid prompt selection: '{raw_selector}'. "
                "Choose from the numbered prompt list or prompt names."
            )

        if prompt_func.__name__ in seen_prompt_names:
            continue

        seen_prompt_names.add(prompt_func.__name__)
        selected_prompt_functions.append(prompt_func)

    if not selected_prompt_functions:
        raise ValueError("Please select at least one prompt.")

    return selected_prompt_functions


def select_prompt_functions():
    available_prompts = get_available_prompts()
    if not available_prompts:
        raise ValueError("No prompts are currently defined in Prompts.py.")

    print("Available prompts:")
    for idx, prompt_func in enumerate(available_prompts, start=1):
        print(f"{idx}. {get_prompt_display_name(prompt_func)}")

    while True:
        try:
            raw_selection = input(
                "Select prompt number(s) or name(s), separated by commas: "
            ).strip()
        except EOFError as exc:
            raise ValueError("Prompt selection was not provided.") from exc

        if not raw_selection:
            print("Please enter at least one prompt number or name.")
            continue

        try:
            return parse_prompt_selection(raw_selection, available_prompts)
        except ValueError as exc:
            print(str(exc))


def get_selected_prompt_functions(prompt_selection_enabled, prompt_selectors=None):
    available_prompts = get_available_prompts()
    if not available_prompts:
        raise ValueError("No prompts are currently defined in Prompts.py.")

    if prompt_selectors:
        return parse_prompt_selection(prompt_selectors, available_prompts)

    if prompt_selection_enabled:
        return select_prompt_functions()

    return [Safety_Prompt]

'''
Example CLI usage
python Unit2/NHTSA_Project/NHTSA_Project_Main.py --samples 100 --random-sampling --prompts Specific_Subsystem_Prompt
'''
# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="NHTSA Complaint Analysis")
    parser.add_argument(
        "--output", 
        default="Unit2/NHTSA_Project/Outputs", 
        required=False,
        help="Directory to store CSV outputs (default: Unit2/NHTSA_Project/Outputs)"
    )
    parser.add_argument(
        "--chart", 
        required=False,
        help="Function name in Create_Charts to call (e.g., create_bar_chart)"
    )
    parser.add_argument(
        "--chart-csv", 
        required=False,
        help="Path to the CSV file to create a chart from"
    )
    parser.add_argument(
        "--columns", 
        nargs="+",
        required=False,
        help="Column(s) to use for the chart"
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=10,
        required=False,
        help="Number of samples to process from the NHTSA database. Can be an integer, 'All', or 'None'. Defaults to 10 unless --random-sampling is used."
    )
    parser.add_argument(
        "--random-sampling",
        action="store_true",
        help="Randomly sample unprocessed complaints without replacement. Requires --samples."
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="Interactively choose one or more predefined prompts from Prompts.py."
    )
    parser.add_argument(
        "--prompts",
        type=str,
        required=False,
        help="Comma-separated prompt numbers or names, e.g. '1,2' or 'Safety_Prompt,Specific_Subsystem_Prompt'."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        required=False,
        help=(
            "Number of parallel worker threads for LLM calls. "
            "Each worker makes its own API call concurrently. "
            "If not provided, all available system workers are used, "
            "which can cause the system to lag. "
            "Decrease if you hit API rate limits."
        )
    )

    args = parser.parse_args()
    
    if args.chart:
        if not args.chart_csv:
            parser.error("--chart requires --chart-csv.")
        if not args.columns:
            parser.error("--chart requires --columns.")

    if args.random_sampling and args.samples is None:
        parser.error("--random-sampling requires --samples.")

    if args.prompts:
        try:
            parse_prompt_selection(args.prompts, get_available_prompts())
        except ValueError as exc:
            parser.error(str(exc))

    try:
        resolve_num_samples(args.samples, random_sampling=args.random_sampling)
    except ValueError as exc:
        parser.error(str(exc))
            
    return args

import Project_Tools.Create_Charts as Create_Charts

def main():
    args = get_args()
    
    if args.chart:
        if hasattr(Create_Charts, args.chart):
            chart_func = getattr(Create_Charts, args.chart)
            chart_func(args.chart_csv, args.columns, args.output)
        else:
            print(f"Error: Function '{args.chart}' not found in Create_Charts.py")
        return
    
    df = load_df()
    print(f"Loaded {len(df):,} rows x {len(df.columns)} columns")

    prompt_functions = get_selected_prompt_functions(args.prompt, args.prompts)
    complaint_column = get_complaint_column(df)

    print(
        "Using prompt(s): "
        + ", ".join(get_prompt_display_name(prompt_func) for prompt_func in prompt_functions)
    )

    processed_indices_by_prompt = {
        prompt_func.__name__: get_processed_df_indices(args.output, prompt_func.__name__)
        for prompt_func in prompt_functions
    }

    for prompt_func in prompt_functions:
        processed_count = len(processed_indices_by_prompt[prompt_func.__name__])
        if processed_count:
            print(
                f"Found {processed_count:,} previously processed row(s) in "
                f"{prompt_func.__name__}.csv"
            )

    processed_indices = set()
    for prompt_func in prompt_functions:
        processed_indices.update(processed_indices_by_prompt[prompt_func.__name__])

    num_samples = resolve_num_samples(
        args.samples,
        random_sampling=args.random_sampling,
    )
    if num_samples is None:
        num_samples = len(df)

    if args.random_sampling:
        rows_to_process = get_random_sample_rows(
            df=df,
            n=num_samples,
            processed_indices=processed_indices,
        )
        print(
            f"Randomly selected {len(rows_to_process)} unprocessed sample(s) "
            "without replacement."
        )
    else:
        rows_to_process = get_sequential_rows(
            df=df,
            n=num_samples,
            processed_indices=processed_indices,
        )
        print(
            f"Processing {len(rows_to_process)} sequential unprocessed sample(s)..."
        )

    if rows_to_process.empty:
        print("No unprocessed rows available for this prompt.")
        return

    total_samples = len(rows_to_process)
    worker_label = args.workers if args.workers is not None else "all available"
    print(f"Dispatching {total_samples} row(s) across {worker_label} worker thread(s)...")

    def process_one(df_index, row):
        """
        Worker function: processes one complaint row for every selected prompt.

        Runs in a thread — all shared state it reads (prompt_functions,
        processed_indices_by_prompt, complaint_column, args) is either
        read-only or updated exclusively from the main thread after futures
        complete, so no additional locking is needed here.
        """
        complaint = str(row[complaint_column])
        results = {}  # prompt_func_name -> parsed_json (or None on failure)

        print(f"\n=== Complaint index {df_index} ===")

        for prompt_func in prompt_functions:
            # Redundant safety check — rows_to_process was already pre-filtered,
            # but this guards against edge cases where a row slips through.
            if df_index in processed_indices_by_prompt[prompt_func.__name__]:
                print(
                    f"Skipping {df_index} for {get_prompt_display_name(prompt_func)} "
                    "because it is already stored in that task's CSV."
                )
                continue

            print(f"Using {get_prompt_display_name(prompt_func)}")

            prompts = prompt_func(complaint)
            messages = LLM_Message(system_prompt=prompts["system"], user_prompt=prompts["user"])
            tools = LLM_Tools()

            parsed_json, _ = process_and_store_response(
                call_llm_func=Call_LLM,
                messages=messages,
                tools=tools,
                df_index=df_index,
                prompt_func_name=prompt_func.__name__,
                output_dir=args.output,
            )

            results[prompt_func.__name__] = parsed_json

        return df_index, results

    # Submit all rows to the thread pool. Each future represents one complaint row.
    # LLM calls are I/O-bound (network), so threading is the right primitive here —
    # threads yield the GIL while waiting on the socket, so workers run truly in parallel.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, df_index, row): df_index
            for df_index, row in rows_to_process.iterrows()
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            submitted_index = futures[future]

            try:
                df_index, results = future.result()
            except Exception as e:
                # An unexpected exception (not a handled server error) escaped
                # process_and_store_response. Log it and continue so one bad row
                # doesn't abort the entire batch.
                print(
                    f"Unexpected error in worker for index {submitted_index}: {e}. "
                    "Skipping this row."
                )
                continue

            # Update processed_indices_by_prompt from the main thread only —
            # avoids any concurrent mutation of the shared sets.
            for func_name, parsed_json in results.items():
                if parsed_json:
                    processed_indices_by_prompt[func_name].add(df_index)
                    prompt_display = next(
                        get_prompt_display_name(pf)
                        for pf in prompt_functions
                        if pf.__name__ == func_name
                    )
                    print(
                        f"Final Parsed Output for {df_index} ({prompt_display}):",
                        parsed_json,
                    )
                else:
                    print(
                        f"Failed to process complaint at index {df_index} "
                        f"with {func_name}."
                    )

            print(f"Progress: {completed}/{total_samples} complete")


if __name__ == "__main__":
    main()
