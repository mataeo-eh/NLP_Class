import pandas as pd
from pathlib import Path

import os
import litellm
import argparse
from litellm import completion
from dotenv import load_dotenv

from Prompts import Specific_Subsystem_Prompt, Safety_Prompt




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
file_path = Path("NHTSA/complaints_cleaned.parquet")


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

    temperature=1,
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

from Project_Tools.Store_LLM_Responses import process_and_store_response

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
        default="10",
        required=False,
        help="Number of samples to process from the NHTSA database. Can be an integer, 'All', or 'None'. Default is 10."
    )
    
    args = parser.parse_args()
    
    if args.chart:
        if not args.chart_csv:
            parser.error("--chart requires --chart-csv.")
        if not args.columns:
            parser.error("--chart requires --columns.")
            
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
    
    if args.samples.lower() in ["all", "none"]:
        num_samples = len(df)
    else:
        try:
            num_samples = int(args.samples)
        except ValueError:
            print(f"Invalid --samples value: '{args.samples}'. Defaulting to 10.")
            num_samples = 10
            
    num_samples = min(num_samples, len(df))
    print(f"Processing {num_samples} sample(s)...")
    
    for row_pos in range(num_samples):
        df_index = df.index[row_pos]
        complaint = str(df.iloc[row_pos, 19])
        print(f"\n--- Processing complaint at index {df_index} ({row_pos + 1}/{num_samples}) ---")
        
        # Use Safety_Prompt (as an example)
        prompt_func = Safety_Prompt
        prompts = prompt_func(complaint)
        
        messages = LLM_Message(system_prompt=prompts["system"], user_prompt=prompts["user"])
        tools = LLM_Tools()
        
        # Process and store, passing the Call_LLM function reference and the true df_index
        parsed_json, raw_content = process_and_store_response(
            call_llm_func=Call_LLM,
            messages=messages,
            tools=tools,
            df_index=df_index,
            prompt_func_name=prompt_func.__name__,
            output_dir=args.output
        )
        
        if parsed_json:
            print(f"Final Parsed Output for {df_index}:", parsed_json)
        else:
            print(f"Failed to process complaint at index {df_index}.")


if __name__ == "__main__":
    main()