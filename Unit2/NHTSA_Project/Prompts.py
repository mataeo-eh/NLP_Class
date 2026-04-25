import sys
import os
import pandas as pd
from pathlib import Path
import inspect
from pprint import pprint

# ---------------------------------------------------------------------------
# Import schema type constants from Build_DF.py so build_schema_summary()
# can annotate columns with their correct data type (numeric vs. date vs. text).
# COLUMN_NAMES is not imported here because the descriptions are maintained as
# a dict below — the raw list from Build_DF has no runtime-accessible descriptions.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "NHTSA"))
from NHTSA.Build_DF import NUMERIC_COLS, DATE_COLS

# ---------------------------------------------------------------------------
# Column descriptions for the 49-column NHTSA complaints Parquet database.
# Derived from the NHTSA CMPL.txt data dictionary (same source as Build_DF.py).
# Maintained here rather than parsed from Build_DF comments so they are
# available at runtime as a string for injection into the retrieval prompt.
# ---------------------------------------------------------------------------
_COLUMN_DESCRIPTIONS = {
    "CMPLID":            "NHTSA internal unique sequence number",
    "ODINO":             "NHTSA internal reference number",
    "MFR_NAME":          "Manufacturer's name",
    "MAKETXT":           "Vehicle/equipment make",
    "MODELTXT":          "Vehicle/equipment model",
    "YEARTXT":           "Model year (9999 if unknown or N/A)",
    "CRASH":             "Was vehicle involved in a crash (Y/N)",
    "FAILDATE":          "Date of incident (YYYYMMDD)",
    "FIRE":              "Was vehicle involved in a fire (Y/N)",
    "INJURED":           "Number of persons injured",
    "DEATHS":            "Number of fatalities",
    "COMPDESC":          "Component category (list-typed — one complaint can span multiple categories)",
    "CITY":              "Consumer's city",
    "STATE":             "Consumer's state code",
    "VIN":               "Vehicle identification number",
    "DATEA":             "Date complaint was added to the NHTSA file (YYYYMMDD)",
    "LDATE":             "Date complaint was received by NHTSA (YYYYMMDD)",
    "MILES":             "Vehicle mileage at time of failure",
    "OCCURENCES":        "Number of occurrences of the described failure",
    "CDESCR":            "Full text description of the complaint",
    "CMPL_TYPE":         "Source of complaint code",
    "POLICE_RPT_YN":     "Was incident reported to police (Y/N)",
    "PURCH_DT":          "Date vehicle was purchased (YYYYMMDD)",
    "ORIG_OWNER_YN":     "Was complainant the original owner (Y/N)",
    "ANTI_BRAKES_YN":    "Vehicle equipped with anti-lock brakes (Y/N)",
    "CRUISE_CONT_YN":    "Vehicle equipped with cruise control (Y/N)",
    "NUM_CYLS":          "Number of engine cylinders",
    "DRIVE_TRAIN":       "Drive train type (AWD/4WD/FWD/RWD)",
    "FUEL_SYS":          "Fuel system code (FI/TB)",
    "FUEL_TYPE":         "Fuel type code (BF/CN/DS/GS/HE)",
    "TRANS_TYPE":        "Transmission type (AUTO/MAN)",
    "VEH_SPEED":         "Vehicle speed at time of incident",
    "DOT":               "DOT tire identifier",
    "TIRE_SIZE":         "Tire size",
    "LOC_OF_TIRE":       "Location of tire code",
    "TIRE_FAIL_TYPE":    "Type of tire failure code",
    "ORIG_EQUIP_YN":     "Was the part original equipment (Y/N)",
    "MANUF_DT":          "Date of manufacture (YYYYMMDD)",
    "SEAT_TYPE":         "Type of child seat code",
    "RESTRAINT_TYPE":    "Child seat installation system code",
    "DEALER_NAME":       "Dealer's name",
    "DEALER_TEL":        "Dealer's telephone number",
    "DEALER_CITY":       "Dealer's city",
    "DEALER_STATE":      "Dealer's state code",
    "DEALER_ZIP":        "Dealer's zip code",
    "PROD_TYPE":         "Product type code (V=vehicle / T=tire / E=equipment / C=child seat)",
    "REPAIRED_YN":       "Was the defective tire repaired (Y/N)",
    "MEDICAL_ATTN":      "Was medical attention required (Y/N)",
    "VEHICLES_TOWED_YN": "Was the vehicle towed (Y/N)",
}

# ---------------------------------------------------------------------------
# NHTSA Component Taxonomy
# ---------------------------------------------------------------------------
# Programmatically derived from the COMPDESC field in the complaints data file.
# Each COMPDESC entry follows a hierarchical format: TOP_LEVEL:SUB:SUB...
# We extract unique top-level categories (the part before the first ':') and
# filter to entries that match NHTSA's ALL-CAPS coding convention, excluding
# free-text noise that occasionally appears in the raw data.
# ---------------------------------------------------------------------------
def _load_nhtsa_taxonomy() -> list[str]:
    """
    Read the complaints file and return a sorted list of unique top-level
    NHTSA component categories. Filters out free-text entries (those containing
    lowercase letters) which are not part of the official taxonomy.
    """
    data_file = Path(__file__).parent / "NHTSA" / "COMPLAINTS_RECEIVED_2025-2026.txt"
    try:
        df = pd.read_csv(
            data_file,
            sep="\t",
            header=None,
            usecols=[11],        # COMPDESC is the 12th column (0-based index 11)
            dtype=str,
            encoding="latin-1",  # NHTSA files use latin-1 encoding
        )
        df.columns = ["COMPDESC"]

        # Split on ':' and take the first segment as the top-level category
        top_level = (
            df["COMPDESC"]
            .dropna()
            .str.split(":").str[0]
            .str.strip()
        )

        # Keep only entries that follow NHTSA's ALL-CAPS convention —
        # this filters out free-text complaint descriptions that leaked
        # into the COMPDESC field (e.g., "I suspect the car seat is counterfeit")
        taxonomy = sorted({
            val for val in top_level.unique()
            if len(val) >= 3
            and val == val.upper()  # All-uppercase signals a proper NHTSA code
            and not val.isdigit()
        })
        return taxonomy
    except Exception:
        # If the data file is unavailable, return an empty list so the
        # prompt degrades gracefully rather than crashing at import time
        return []

NHTSA_COMPONENT_TAXONOMY: list[str] = _load_nhtsa_taxonomy()


def get_available_prompts():
    prompt_functions = []

    for obj in globals().values():
        if not inspect.isfunction(obj):
            continue
        if obj.__module__ != __name__:
            continue
        if obj.__name__.startswith("_"):
            continue
        if not obj.__name__.endswith("_Prompt"):
            continue

        prompt_functions.append(obj)

    return prompt_functions


def get_prompt_display_name(prompt_func):
    prompt_name = prompt_func.__name__
    if prompt_name.endswith("_Prompt"):
        prompt_name = prompt_name[:-7]

    return f"{prompt_name.replace('_', ' ')} Prompt"


complaint = None
def Safety_Prompt(complaint):
    system_prompt = f'''
You are a highly analytical vehicle safety expert.

You must return your response STRICTLY as a valid JSON object containing exactly the four fields described below. Do not include any text, markdown formatting, or commentary outside of this JSON object.

### Output JSON Structure:
1. "safety_category": A descriptive category for the safety issue. You may use previously established categories (provided in previous Assistant messages, e.g., 'fire hazard', 'seatbelt safety', 'driving safety', 'road conditions') or formulate a new, fitting category if the complaint presents a novel issue.
2. "level_of_danger": The estimated likelihood that the complaint will lead to fatalities if unaddressed. You must select exactly one of the following, in order of increasing severity: 'None', 'Low', 'mild', 'Moderate', 'High', or 'Severe'.
3. "urgency_for_human_review": How urgently the manufacturer needs to review this complaint. You must select exactly one of the following: 'Low', 'Urgent', or 'Emergent'.
4. "reasoning_and_justification": A thorough, analytical explanation of your choices for the category, danger level, and urgency. All of your rationale, including statistical reasoning or specific complaint details, MUST go here.

### Rules for Urgency Classification:
- **Emergent Triggers:**
  - Any complaint with a 'Severe' level of danger MUST be classified as 'Emergent'.
  - Any complaint involving a 'fire hazard', potential combustion, or related risks MUST be classified as 'Emergent', regardless of its level of danger.
  - **High Volume:** If the assistant role messages indicate a significantly high percentage of previous complaints fall under the same category (e.g., 500 out of 1000 complaints are 'driving safety'), you must elevate the urgency to 'Emergent' due to the widespread nature of the issue.
- **Lower Urgency Exceptions:** Complaints regarding primarily cosmetic issues (e.g., a piece of trim flying off on the highway) should generally be given lower urgency for human review. While such issues might carry a higher "level of danger" due to the risk of flying debris at high speeds, they are typically less urgent for the manufacturer to review immediately compared to systemic mechanical failures.

Rely on your expert judgment to provide a comprehensive, naturally articulated reasoning in the "reasoning_and_justification" field. Include any relevant details regarding previous complaint percentages only if they warrant increasing the human review urgency.
'''

    user_prompt = f'''
Given the user complaint below, conduct a deep-reasoning evaluation to identify safety concerns.

Perform your analysis independently, exploring the nuances of the complaint to articulate a unique, well-considered justification for your classifications.

--- BEGIN USER COMPLAINT ---
{complaint}
--- END USER COMPLAINT ---
'''
    return {"system": system_prompt, "user": user_prompt}

def Specific_Subsystem_Prompt(complaint, taxonomy: list[str] = NHTSA_COMPONENT_TAXONOMY):
    # This prompt asks the LLM to classify which NHTSA component category (or categories)
    # a complaint belongs to, using the standardized taxonomy derived directly from the
    # complaints data file. Multiple labels are allowed — a single complaint can legitimately
    # span more than one component category (e.g., both "ELECTRICAL SYSTEM" and
    # "VEHICLE SPEED CONTROL"). The LLM output can later be compared against the actual
    # NHTSA COMPDESC field as a curiosity — not a formal accuracy metric.
    # In production, COMPDESC is read directly from the database rather than inferred by the LLM.

    # Format the taxonomy as a numbered list for the system prompt
    taxonomy_list = "\n".join(f"  {i+1}. {label}" for i, label in enumerate(taxonomy))

    system_prompt = f'''
You are a vehicle safety analyst. Your task is to classify which NHTSA component category or categories a complaint belongs to.

You MUST select labels exclusively from the standardized NHTSA taxonomy listed below. Do not invent new labels.
A complaint may fit more than one category simultaneously — assign every label that applies.

### NHTSA Component Taxonomy (select one or more):
{taxonomy_list}

Return ONLY a valid JSON object. Do not include any text, explanation, or markdown outside the JSON.

The JSON must have exactly these three fields:
{{
  "subsystems": ["<NHTSA label>", "<NHTSA label>"],
  "confidence": "<Low | Medium | High>",
  "reasoning": "<one or two sentences explaining which complaint details led to each assigned label>"
}}

Rules:
- "subsystems" must be a JSON array containing at least one label.
- Every label in "subsystems" must be copied verbatim from the taxonomy above.
- If the complaint clearly spans multiple categories, include all that apply.
- If the complaint does not fit any specific category, use "UNKNOWN OR OTHER".
'''

    user_prompt = f'''
Given the vehicle owner complaint below, identify which NHTSA component category or categories it belongs to.
Assign every label from the taxonomy that applies — do not limit yourself to one if multiple fit.

--- BEGIN USER COMPLAINT ---
{complaint}
--- END USER COMPLAINT ---
'''
    return {"system": system_prompt, "user": user_prompt}


def Specify_Task_Type(user_request: str):
    # Routes a user request into one of three task types that control how the LangGraph
    # pipeline behaves downstream. The distinction that matters most:
    #
    #   "analyze"                   — the data to reason over is pre-determined by scripting
    #                                 (e.g., complaint text already in state). The model only
    #                                 analyzes; it does not decide what to fetch.
    #
    #   "retrieve"                  — a pure data lookup with no analysis. The user wants
    #                                 specific records or fields returned, nothing more.
    #
    #   "agentic_retrieve_and_analyze" — the model must drive its own multi-step retrieval
    #                                 based on intermediate analysis results. Retrieval
    #                                 strategy is NOT pre-determined; the model decides what
    #                                 to fetch next based on what it finds.
    #
    # The model must return exactly one of those three strings and nothing else.
    # This output populates State["task_type"] and gates every subsequent node.

    system_prompt = '''\
You are a task router for an NHTSA vehicle safety complaint analysis pipeline.

Your only job is to read the user's request and classify it into exactly one of three task types:

  analyze
  retrieve
  agentic_retrieve_and_analyze

### Definitions

analyze
  The information to be reasoned over is already determined by the pipeline — it will be
  fetched by scripting before the model sees it. The model's job is purely to reason,
  classify, label, or evaluate that pre-fetched data. The model does NOT decide what to
  retrieve. No additional data lookup is driven by the model's own intermediate conclusions.
  Example: "Classify the subsystem components this complaint describes."
  Example: "Rate the danger level of this complaint."

retrieve
  The user wants specific data fetched and returned. No analysis, labeling, or reasoning
  is needed — just find the records or field values and report them.
  Example: "How many complaints involve Toyota Camry brake failures?"
  Example: "What is the COMPDESC for complaint ID 12345?"

agentic_retrieve_and_analyze
  The request requires the model to analyze something, then use the results of that analysis
  to decide what additional data to retrieve, then revise or extend its analysis based on
  what it retrieved. The retrieval strategy is NOT pre-determined — the model drives it
  dynamically based on its own intermediate conclusions. This is a multi-step loop where
  analysis and retrieval alternate.
  Example: "Analyze this complaint for danger level, then pull injury/death counts for
            similar complaints, and re-evaluate your danger rating with that context."
  Example: "Classify the complaint, then retrieve recall data for the affected component,
            then summarize the combined picture."

### Output Rule

Respond with EXACTLY one of these three strings and nothing else — no punctuation, no
explanation, no reasoning, no newline, no quotes:

  analyze
  retrieve
  agentic_retrieve_and_analyze

Any response that contains more than one of those strings, or contains any other text
whatsoever, is wrong.
'''

    user_prompt = f'''\
Classify the following user request into exactly one task type.

Respond with only one of: analyze, retrieve, agentic_retrieve_and_analyze

--- BEGIN USER REQUEST ---
{user_request}
--- END USER REQUEST ---
'''

    return {"system": system_prompt, "user": user_prompt}


# ---------------------------------------------------------------------------
# Schema summary builder — used by Retrieve_Data_Prompt to inject the full
# column reference into the retrieval system prompt at call time.
# ---------------------------------------------------------------------------

def build_schema_summary() -> str:
    # Format each of the 49 complaint columns as a single annotated line.
    # Type annotation comes from NUMERIC_COLS / DATE_COLS imported at the top;
    # everything else defaults to "text".
    lines = [
        "NHTSA Complaints Database — Column Reference",
        "=" * 60,
        f"{'Column':<22} {'Type':<10} Description",
        "-" * 60,
    ]

    for col, desc in _COLUMN_DESCRIPTIONS.items():
        if col in NUMERIC_COLS:
            col_type = "integer"
        elif col in DATE_COLS:
            col_type = "date"
        else:
            col_type = "text"
        lines.append(f"  {col:<20} {col_type:<10} {desc}")

    lines.append("")
    lines.append(
        "Note: COMPDESC is list-typed. A single complaint can span multiple component "
        "categories. When filtering on COMPDESC, use the component name as the value "
        "and the filter will check whether it appears anywhere in the list."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retrieval prompt — drives the retrieve_data node's agentic mini-loop
# ---------------------------------------------------------------------------

def Retrieve_Data_Prompt(user_request: str, schema_str: str) -> dict:
    # This prompt is the system context for the retrieve_data node in the LangGraph
    # pipeline. The node runs a bounded agentic loop (up to 8 iterations) where the
    # LLM calls tools to query data, clarify ambiguity, and ultimately return a
    # structured result as its final message content.
    #
    # Key responsibilities given to the model:
    #   1. Decide which data source to query (parquet complaints DB vs. CSV outputs)
    #   2. Reason about how many results to return and which fields are wanted
    #   3. Ask the user for clarification when the request is genuinely ambiguous
    #   4. Cap results at 10 rows regardless of user request
    #   5. When done, produce a final natural-language + data response as message
    #      content (no trailing tool call) so the loop can exit cleanly

    system_prompt = f'''\
You are a data retrieval assistant for an NHTSA vehicle safety complaint analysis pipeline.
Your job is to fulfill the user's data request by querying the available data sources using
the tools provided to you.

You are operating inside a bounded loop. When you are finished retrieving and formatting
the data, return your final answer as plain message content — do NOT make another tool call.
The loop exits as soon as you produce a response with no tool calls.

===============================================================================
DATA SOURCE 1 — NHTSA Complaints Parquet Database (~90,000 rows)
===============================================================================
Use get_rows_by_position and filter_rows to access this database.

{schema_str}

===============================================================================
DATA SOURCE 2 — Pipeline CSV Output Files (Outputs directory)
===============================================================================
The pipeline writes analysis results to CSV files in the Outputs directory. These
files contain LLM-generated labels, safety scores, subsystem classifications,
reasoning text, and comparisons against human-assigned labels.

To work with CSV output files:
  1. Call list_csv_files() to see what files are available.
  2. Call get_csv_schema(filename) to understand a file's column structure before
     querying it — this is mandatory if you have not yet inspected the file.
  3. Use filter_csv() or get_csv_rows_by_position() to retrieve the data.

===============================================================================
RESULT CAP
===============================================================================
You may return at most 10 rows per response. If the user requests more than 10,
cap at 10 and note this in your response.

===============================================================================
REASONING BEFORE FETCHING — REQUIRED
===============================================================================
Before executing any query, think carefully about what the user actually wants.

HOW MANY RESULTS?
  - If the user specifies a number, use that (capped at 10).
  - If the request is clearly a single-record lookup (e.g. "what is the complaint
    for row 4587", "show me that entry"), fetch 1 result.
  - If the request is ambiguous and could mean multiple results (e.g. "show me
    some brake complaints", "give me examples"), use Ask_User to ask how many
    they want, then call User_Answer() to receive their response before fetching.

WHICH FIELDS?
  - If the user names a specific field (e.g. "what was the complaint text",
    "show me the make and model"), return only those fields.
  - If the user asks to "show me the entry", "show me the record", or uses
    similarly broad language, return all non-NaN fields for each row.
  - If it is genuinely unclear which fields the user wants, use Ask_User to
    ask, then call User_Answer() before fetching.

WHEN TO ASK VS. WHEN TO INFER
  Ask when: getting it wrong would return something useless (e.g. 1 result
  when the user wanted 5, or just one field when they wanted the whole record).
  Infer when: context makes the intent clear even if not stated explicitly.
  Do not ask unnecessary clarifying questions — use your judgment.

===============================================================================
FINAL RESPONSE FORMAT
===============================================================================
When you have retrieved the data, produce a single final response that:
  1. Briefly describes what you retrieved and from which source.
  2. Presents the data clearly (you may format it as a readable list or table in text).
  3. Notes any capping, filtering, or clarification decisions you made.

Do not call any more tools in your final response — the loop exits on the first
message you produce without a tool call.
'''

    user_prompt = f'''\
--- BEGIN USER REQUEST ---
{user_request}
--- END USER REQUEST ---
'''

    return {"system": system_prompt, "user": user_prompt}
