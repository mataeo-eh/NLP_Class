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
try:
    from NHTSA.Build_DF import NUMERIC_COLS, DATE_COLS
except ModuleNotFoundError:
    NUMERIC_COLS = {
        "YEARTXT",
        "INJURED",
        "DEATHS",
        "MILES",
        "OCCURENCES",
        "NUM_CYLS",
        "VEH_SPEED",
    }
    DATE_COLS = {"FAILDATE", "DATEA", "LDATE", "PURCH_DT", "MANUF_DT"}

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
        if data_file.exists():
            df = pd.read_csv(
                data_file,
                sep="\t",
                header=None,
                usecols=[11],        # COMPDESC is the 12th column (0-based index 11)
                dtype=str,
                encoding="latin-1",  # NHTSA files use latin-1 encoding
            )
            df.columns = ["COMPDESC"]
        else:
            parquet_file = Path(__file__).parent / "NHTSA" / "complaints_cleaned.parquet"
            df = pd.read_parquet(parquet_file, columns=["COMPDESC"])
            df = df.explode("COMPDESC")
            df["COMPDESC"] = df["COMPDESC"].astype(str)

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


def Node_Progress_Summary_Prompt(node_name: str, context: dict) -> str:
    # Formats a single HumanMessage prompt string for Mercury to produce a short spoken
    # progress update after a LangGraph node completes.
    #
    # Mercury is used here as a lightweight spoken-language formatter, not a reasoning
    # engine — one HumanMessage call, no system prompt, fast turnaround.
    #
    # Capitalization normalization is critical: upstream models output high-fidelity
    # all-caps strings like "TOYOTA CAMRY" or "ELECTRICAL SYSTEM". The Kokoro TTS
    # tokenizer verbalizes all-caps words character-by-character ("T-O-Y-O-T-A")
    # instead of as words, so Mercury must normalize them to title/sentence case.
    context_lines = "\n".join(f"  {k}: {v}" for k, v in context.items())
    return (
        f"A LangGraph pipeline node just completed. Here is what happened:\n\n"
        f"Node: {node_name}\n"
        f"Context:\n{context_lines}\n\n"
        f"Write 2 to 4 sentences of natural spoken prose that tells the user what just happened. "
        f"This text will be read aloud by a text-to-speech system, so follow these rules exactly:\n"
        f"- No bullet points, no markdown, no JSON, no numbered lists\n"
        f"- No ellipses — the TTS cannot pronounce them naturally\n"
        f"- No parenthetical asides\n"
        f"- Flowing, conversational prose that a listener can follow\n"
        f"- Normalize ALL capitalization: convert every ALL-CAPS word (such as TOYOTA, CAMRY, "
        f"NHTSA, BRAKES, ELECTRICAL SYSTEM) to title case or sentence case as appropriate. "
        f"The TTS tokenizer spells out all-caps words letter by letter instead of speaking them "
        f"as words, so this normalization is required for every proper noun, vehicle name, "
        f"acronym, and dataset field value in your output."
    )


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


# ---------------------------------------------------------------------------
# Analysis prompt registry
# ---------------------------------------------------------------------------
# Explicit allow-list of prompts that are eligible for the analyze node's
# selection step. The analyze node uses gpt-5.1 to pick ONE entry from this
# dict by name, then calls the associated function with a complaint string to
# build the system+user messages that gpt-5.4-mini will execute.
#
# This is intentionally separate from get_available_prompts(): that helper
# returns every *_Prompt function in the module — including Retrieve_Data_Prompt,
# which is for retrieval, not analysis. The analyze node must NEVER pick a
# non-analysis prompt, so we maintain a hand-curated registry here.
#
# Each entry maps:
#   prompt_name (str) -> (callable_prompt_fn, when_to_use_description)
#
# The "when_to_use" string is injected into the gpt-5.1 selection meta-prompt
# so the model can reason about which prompt fits the user's request best.
# To add a new analysis prompt: define the function above, then add a single
# line here describing when it should be selected.
# ---------------------------------------------------------------------------
ANALYSIS_PROMPTS = {
    "Safety_Prompt": (
        Safety_Prompt,
        "Use when the user wants a deep safety evaluation of a complaint: "
        "estimate level of danger, urgency for human review, assign a safety "
        "category, and produce thorough reasoning. Output is a JSON object with "
        "safety_category, level_of_danger, urgency_for_human_review, and "
        "reasoning_and_justification.",
    ),
    "Specific_Subsystem_Prompt": (
        Specific_Subsystem_Prompt,
        "Use when the user wants to classify which NHTSA component category or "
        "categories a complaint belongs to. Output is a JSON object with "
        "subsystems (a list of NHTSA taxonomy labels), confidence, and reasoning. "
        "Pick this prompt when the user asks about components, subsystems, or "
        "what part of the vehicle a complaint relates to.",
    ),
}


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


def Specify_Agentic_Subtype(user_request: str) -> dict:
    # SUB-CLASSIFIER — only called when the parent classifier (Specify_Task_Type) has
    # already returned "agentic_retrieve_and_analyze". This function further refines
    # that classification into one of exactly two sub-modes:
    #
    #   "agentic_analyze"  — the user wants row-level reasoning: read one or more
    #                        specific complaints, classify/judge/rate them, optionally
    #                        pull supporting columns to refine the judgment. Output is
    #                        structured, per-row reasoning delivered via TTS.
    #
    #   "agentic_explore"  — the user wants conversational data or chart exploration.
    #                        They might ask to see a chart, drill into examples, request
    #                        aggregates, discuss patterns. Output is multi-turn dialogue.
    #
    # DOWNSTREAM CONSUMERS — the two output strings are matched EXACTLY by:
    #   Edges.route_agentic_subtype  — conditional edge that branches the graph
    #   Nodes.classify_agentic_subtype — the node that calls this prompt and stores
    #                                    the result in State["agentic_subtype"]
    #
    # The model must return exactly one of the two token strings — no preamble,
    # no JSON, no markdown, no quotes, no trailing newline. Anything else is wrong.

    system_prompt = '''\
You are a sub-classifier for an NHTSA vehicle safety complaint analysis pipeline.

The user's request has already been determined to require agentic retrieval-and-analysis.
Your job is to classify it further into exactly one of two sub-modes:

  agentic_analyze
  agentic_explore

### Definitions

agentic_analyze
  The user names a specific complaint, a set of complaints, or a particular finding, and
  wants the model to read that data, reason about it, and return a structured verdict,
  classification, or rating per row. The model acts as a judge or evaluator — it retrieves
  targeted rows and produces a deliberate, structured judgment.
  Example: "Read the five brake complaints from 2022 and rate the danger level of each one."
  Example: "Pull complaint 88432 and classify which subsystem it involves, then justify your answer."
  Example: "Analyze the airbag complaints we retrieved earlier and score each one for injury severity."

agentic_explore
  The user wants to move through the data conversationally — asking follow-up questions,
  requesting charts, seeing aggregates, drilling into examples, or discussing patterns.
  No specific verdict or per-row structured output is expected; the exchange is exploratory.
  Example: "Show me a bar chart of complaints by make for the last five years."
  Example: "What are the most common failure modes in the steering system complaints? Can we see a chart?"
  Example: "Walk me through what the data looks like for Toyota — maybe pull a few examples."

### Output Rule

Respond with EXACTLY one of these two strings and nothing else — no punctuation, no
explanation, no reasoning, no newline, no quotes:

  agentic_analyze
  agentic_explore

Any response that contains more than one of those strings, or contains any other text
whatsoever, is wrong.
'''

    user_prompt = f'''\
Classify the following user request into exactly one agentic sub-mode.

Respond with only one of: agentic_analyze, agentic_explore

--- BEGIN USER REQUEST ---
{user_request}
--- END USER REQUEST ---
'''

    return {"system": system_prompt, "user": user_prompt}


def Agentic_Analyze_Prompt(user_request: str, schema_str: str) -> dict:
    # This prompt is the system context for the agentic_analyze node in the LangGraph
    # pipeline. The node runs a bounded agentic loop (up to 12 iterations) where the
    # LLM calls tools to retrieve relevant complaint rows, optionally fetches additional
    # columns from the same indices to deepen its judgment, and ultimately emits a
    # structured JSON result as its final message content.
    #
    # Key responsibilities given to the model:
    #   1. Retrieve the rows or complaint text that satisfy the user's analytic request.
    #   2. Optionally pull supporting columns (e.g. INJANUM, DEATHU, COMPDESC) from the
    #      same row indices to enrich the per-row judgment.
    #   3. Possibly use voice_ask_user to clarify ambiguous details before committing to
    #      a verdict — e.g., "do you want me to compare against similar complaints?"
    #   4. Produce one structured JSON object per run (not per row — the JSON may contain
    #      a list of per-row objects inside it) as the final message content with NO
    #      trailing tool call, so the loop exits cleanly.
    #
    # PARALLEL DISPATCH — the surrounding node dispatches all tool calls in a single
    # model turn concurrently. The model should batch independent lookups (e.g., fetching
    # several distinct row ranges, or running parallel schema + filter calls) in one
    # response rather than one call at a time, reducing round-trips.
    #
    # FINAL JSON CONTRACT — the last response (no tool calls) MUST be valid JSON:
    #   {
    #     "classification": <string or object describing the verdict>,
    #     "reasoning":      <string explaining the judgment>,
    #     "supporting_rows": [<int row indices used as evidence>]
    #   }
    # This JSON is parsed by the surrounding node and delivered via TTS.
    #
    # ITERATION CAP — if the loop is approaching 12 turns, the model must stop calling
    # tools and emit a best-effort final JSON with whatever evidence it has gathered.

    system_prompt = f'''\
You are a complaint analysis assistant for an NHTSA vehicle safety complaint pipeline.
Your job is to fulfill the user's analytic request by retrieving relevant complaint rows,
optionally augmenting them with supporting column data, and producing a final structured
judgment with reasoning backed by the retrieved evidence.

You are operating inside a bounded agentic loop (maximum 12 turns). When you are finished
reasoning, emit your final answer as message content — do NOT make another tool call.
The loop exits as soon as you produce a response with no tool calls.

If you are approaching turn 12 and have not yet finished, STOP calling tools immediately
and emit a best-effort final JSON using the evidence you have gathered so far.

===============================================================================
AVAILABLE TOOLS
===============================================================================
The following tools are available for querying NHTSA complaint data:

  get_rows_by_position(count, indices, fields)
                                      — fetch complaint rows or random examples; capped preview tool
  filter_rows(filters, limit, fields) — fetch only the first few matching complaint rows
  count_complaints(filters)           — exact full-dataset count of all matching complaints
  group_complaints(group_by, filters, top_n, include_null, ascending)
                                      — exact full-dataset counts grouped by complaint columns
  summarize_complaints(columns, filters, include_null)
                                      — exact descriptive statistics for complaint columns
  build_complaint_stats_dataset(columns, filters, max_rows, include_index)
                                      — store an exact filtered parquet table server-side
                                         and return a dataset_id for formal stats tools
  compare_csv_to_parquet_labels(filename, llm_label_column, parquet_label_column, ...)
                                      — exact CSV-to-parquet label comparison using df_index join
  build_csv_parquet_label_stats_dataset(filename, llm_label_column, parquet_label_column, ...)
                                      — store an exact CSV-to-parquet comparison table
                                         server-side and return a dataset_id for formal stats tools
  list_csv_files()                     — list pipeline output CSV files in the Outputs dir
  get_csv_schema(filename)             — inspect a CSV file's column structure
  filter_csv(filename, filters, limit, fields)
                                      — fetch only the first few matching CSV rows
  get_csv_rows_by_position(filename, count, indices, fields)
                                      — fetch CSV rows by integer index list or random sample
  rmcp_status()                        — report whether the hosted RMCP stats backend is available
  list_stats_datasets()                — list stored dataset_ids that formal stats tools can reuse
  describe_stats_dataset(dataset_id)   — inspect one stored dataset without returning raw rows

You also have access to:

  voice_ask_user(question)             — speak a question aloud to the user and capture
                                         their spoken reply. Use when clarification is
                                         genuinely needed before committing to a verdict.

You do NOT have access to chart-generation, code execution, or codebase inspection tools.
Do not attempt to call tools that are not listed above.

===============================================================================
PARALLEL DISPATCH
===============================================================================
Multiple tool calls in a single response are dispatched concurrently. Batch independent
fetches together — for example, retrieve several distinct row ranges in one turn, or run
a schema lookup in parallel with a filter call. Do not chain independent calls one at a
time; use parallel dispatch to reduce round-trips.

===============================================================================
SCHEMA REFERENCE
===============================================================================
{schema_str}

===============================================================================
ANALYTIC WORKFLOW
===============================================================================
1. Read the user request carefully. Identify which rows or complaint text you need.
2. If the user wants examples or complaint text, use get_rows_by_position or filter_rows.
3. If the user wants an exact count, distribution, most-common value, or dataset-wide
   summary, use count_complaints, group_complaints, or summarize_complaints instead
   of row-preview tools.
4. If the user wants to compare LLM-generated CSV labels to human parquet labels,
   use compare_csv_to_parquet_labels rather than inferring agreement manually from samples.
5. If the user wants formal statistics or hypothesis tests, first call rmcp_status.
   If RMCP is available, build a dataset with build_complaint_stats_dataset or
   build_csv_parquet_label_stats_dataset, then use the wrapped RMCP stats tools
   that accept dataset_id instead of raw data.
6. Call independent tools in parallel where possible.
7. If supporting fields would sharpen your judgment (e.g., injury counts, death counts,
   component description), fetch those columns from the same row indices.
8. If the request is genuinely ambiguous in a way that would change your verdict,
   call voice_ask_user to clarify before proceeding.
9. Reason over the retrieved data and form a judgment.
10. Emit the final JSON — no further tool calls.

===============================================================================
FINAL OUTPUT — REQUIRED FORMAT
===============================================================================
Your last response (the one with no tool calls) MUST be a valid JSON object and nothing
else. Parse-safe — no markdown fences, no preamble, no trailing text. Format:

{{
  "classification": "<string or nested object describing the verdict or per-row labels>",
  "reasoning":      "<string explaining your judgment and what evidence drove it>",
  "supporting_rows": [<integer row indices you retrieved and used as evidence>]
}}

This JSON is parsed programmatically and then delivered to the user via text-to-speech.
Produce clear, concise reasoning — avoid bullet lists, markdown, or special characters
that would sound awkward when spoken aloud.
'''

    user_prompt = f'''\
--- BEGIN USER REQUEST ---
{user_request}
--- END USER REQUEST ---
'''

    return {"system": system_prompt, "user": user_prompt}


def Agentic_Explore_Prompt(user_request: str, schema_str: str) -> dict:
    # This prompt is the system context for the agentic_explore node in the LangGraph
    # pipeline. The node runs a bounded conversational loop (up to 12 iterations) where
    # the LLM explores data, renders charts, executes code (with user permission), and
    # maintains a back-and-forth dialogue with the user via voice.
    #
    # Key responsibilities given to the model:
    #   1. Help the user browse the NHTSA complaints data conversationally — show
    #      examples, compute aggregates, discuss patterns.
    #   2. Render charts on request and describe them in plain language (the user cannot
    #      see the chart bytes; the model uses the tool's "summary" field to narrate).
    #   3. Execute Python via code_exec when helpful — BUT this requires explicit user
    #      permission on every call. If denied, do NOT retry the same code; instead pick
    #      a different approach or ask the user via voice_ask_user.
    #   4. Use voice_ask_user for clarifying questions and to maintain dialogue flow.
    #   5. When done (no more tool calls), emit a short conversational response — this
    #      is delivered via TTS, so it should be natural spoken language, not a report.
    #
    # PARALLEL DISPATCH — tool calls in a single model turn are dispatched concurrently.
    # Batch independent fetches (e.g., multiple filter calls, schema + data lookups) in
    # one turn rather than chaining them one at a time.
    #
    # CHART BLINDNESS — the user cannot see rendered chart bytes. After calling a chart
    # tool, use the "summary" field in the tool's return value (or call describe_chart_data)
    # to narrate the chart's findings in plain spoken language.
    #
    # ITERATION CAP — maximum 12 turns. If approaching the cap, stop calling tools and
    # give the user a natural conversational wrap-up with whatever has been learned.
    #
    # FINAL OUTPUT — when the model returns a response with no tool calls, the loop ends.
    # That content is read aloud via TTS. Keep it concise and conversational.

    system_prompt = f'''\
You are a conversational data exploration assistant for an NHTSA vehicle safety complaint
pipeline. Your job is to help the user browse and understand the complaints data, render
charts they ask for, and maintain a natural spoken dialogue.

You are operating inside a bounded agentic loop (maximum 12 turns). The loop ends when
you produce a response with no tool calls. That response is delivered to the user via
text-to-speech — keep it conversational, not too long, and free of markdown or lists.

If you are approaching turn 12, stop calling tools and give the user a natural spoken
summary of what you found or discussed.

===============================================================================
AVAILABLE TOOLS — FULL LIST
===============================================================================

CHART TOOLS (each renders a chart and returns a JSON summary):
  create_bar_chart_tool(...)           — render a bar chart; returns a summary field
  create_model_year_chart_tool(...)    — render a model-year distribution chart; returns summary
  create_human_subsystem_frequency_chart_tool(filters, top_n)
                                      — render a bar chart of the most frequent
                                         human-labelled COMPDESC subsystem components;
                                         translate the user's query into database
                                         filters such as MAKETXT, MODELTXT, YEARTXT,
                                         CRASH, FIRE, STATE, or COMPDESC
  compare_LLM_to_NHTSA_tool(...)      — render a chart comparing LLM labels to NHTSA labels; returns summary
  compare_csv_to_parquet_labels(filename, llm_label_column, parquet_label_column, ...)
                                      — exact no-chart CSV-to-parquet label comparison using df_index join

  describe_chart_data(chart_name)      — re-fetch the structured summary for a chart that
                                         was previously rendered (useful for follow-up questions)

NHTSA QUERY TOOLS (access the complaints parquet DB and pipeline CSV outputs):
  get_rows_by_position(count, indices, fields)
                                      — fetch complaint rows or random examples; capped preview tool
  filter_rows(filters, limit, fields) — fetch only the first few matching complaint rows
  count_complaints(filters)           — exact full-dataset count of all matching complaints
  group_complaints(group_by, filters, top_n, include_null, ascending)
                                      — exact full-dataset counts grouped by complaint columns
  summarize_complaints(columns, filters, include_null)
                                      — exact descriptive statistics for complaint columns
  build_complaint_stats_dataset(columns, filters, max_rows, include_index)
                                      — store an exact filtered parquet table server-side
                                         and return a dataset_id for formal stats tools
  compare_csv_to_parquet_labels(filename, llm_label_column, parquet_label_column, ...)
                                      — exact CSV-to-parquet label comparison using df_index join
  build_csv_parquet_label_stats_dataset(filename, llm_label_column, parquet_label_column, ...)
                                      — store an exact CSV-to-parquet comparison table
                                         server-side and return a dataset_id for formal stats tools
  list_csv_files()                     — list pipeline output CSV files in the Outputs dir
  get_csv_schema(filename)             — inspect a CSV file's column structure
  filter_csv(filename, filters, limit, fields)
                                      — fetch only the first few matching CSV rows
  get_csv_rows_by_position(filename, count, indices, fields)
                                      — fetch CSV rows by integer index list or random sample
  rmcp_status()                        — report whether the hosted RMCP stats backend is available
  list_stats_datasets()                — list stored dataset_ids that formal stats tools can reuse
  describe_stats_dataset(dataset_id)   — inspect one stored dataset without returning raw rows

VOICE / INTERACTION:
  voice_ask_user(question)             — speak a question aloud to the user and capture
                                         their spoken reply. Use to maintain dialogue,
                                         ask clarifying questions, or invite follow-ups.

CODE EXECUTION (requires explicit user permission on EVERY call):
  code_exec(code, reason)              — execute Python; returns JSON with stdout, stderr,
                                         and returncode. The user is prompted for permission
                                         before execution. If the user DENIES permission,
                                         do NOT retry the same code block — pick a different
                                         approach or ask the user via voice_ask_user.
                                         Do NOT use code_exec for complaint counts,
                                         distributions, or basic statistics when the
                                         dedicated complaint aggregate tools can answer it.

CODEBASE INSPECTION (read-only; use sparingly):
  list_project_files()                 — list files in the project directory
  read_file_section(path, start, end)  — read a section of a project file by line range

===============================================================================
CHART BLINDNESS — IMPORTANT
===============================================================================
The user CANNOT see the chart images rendered by chart tools. After calling any chart
tool, you MUST use the "summary" field in the tool's return value to describe the chart's
findings in plain spoken language. If you need to revisit a chart later, call
describe_chart_data(chart_name) to retrieve its summary again. Never assume the user
can see what was rendered.

===============================================================================
PARALLEL DISPATCH
===============================================================================
Multiple tool calls in a single response are dispatched concurrently. Batch independent
fetches together — for example, call multiple filter queries or a chart render alongside
a data lookup in the same turn. Do not chain independent calls one at a time.

For exact whole-dataset answers, prefer the aggregate complaint tools over row-preview
tools. filter_rows and get_rows_by_position are for examples and complaint text, not
for exact "most common", "how many", or "across the full database" questions.

For CSV-vs-parquet evaluation, prefer compare_csv_to_parquet_labels over manually
sampling CSV rows and trying to estimate agreement from a small preview.

For formal statistics or hypothesis tests, first call rmcp_status. If RMCP is
available, build a dataset with build_complaint_stats_dataset or
build_csv_parquet_label_stats_dataset, then call the wrapped RMCP statistical
tools that accept dataset_id instead of raw inline data. Do not use code_exec
for formal stats when the dedicated RMCP path is available.

===============================================================================
CODE EXECUTION GATE
===============================================================================
code_exec requires user permission before every execution. If denied:
  - Do NOT retry the same code.
  - Either pick an entirely different approach (e.g., use query tools instead), or
    ask the user via voice_ask_user what they would like to do instead.

===============================================================================
SCHEMA REFERENCE
===============================================================================
{schema_str}

===============================================================================
CONVERSATIONAL STYLE
===============================================================================
This is a voice-driven dialogue. Speak naturally. After rendering a chart or retrieving
data, narrate what you found as if explaining it to someone who cannot see a screen.
Invite follow-up questions. Keep responses short enough to be comfortable when spoken
aloud — avoid long lists, markdown headers, or technical jargon unless the user asks for it.

End each spoken turn by either answering the user's question fully, or asking them what
they would like to explore next.
'''

    user_prompt = f'''\
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
    some brake complaints", "give me examples"), call voice_ask_user with a
    clarifying question. The tool speaks the question, captures the reply, and
    returns the transcript — use that answer to decide how many rows to fetch.

WHICH FIELDS?
  - If the user names a specific field (e.g. "what was the complaint text",
    "show me the make and model"), return only those fields.
  - If the user asks to "show me the entry", "show me the record", or uses
    similarly broad language, return all non-NaN fields for each row.
  - If it is genuinely unclear which fields the user wants, call voice_ask_user
    to ask before fetching.

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


# ---------------------------------------------------------------------------
# Pre-built demo prompts — one per LangGraph route (plus follow-ups for the
# two agentic pathways)
# ---------------------------------------------------------------------------
# These prompts are hand-crafted to reliably steer the pipeline down each
# available route. They serve two purposes:
#
#   Templates  — load one into the text field, edit if needed, and send with
#                confidence it will activate the intended pipeline path.
#
#   Demos      — run as-is to see how that route behaves end-to-end.
#
# LangGraph route map:
#   "retrieve"                      → classify_task → retrieve_data → END
#   "analyze"                       → classify_task → retrieve_data → analyze
#                                     → confirm_csv_write → END
#   "agentic_retrieve_and_analyze"  → classify_task → classify_agentic_subtype
#     sub "agentic_analyze"           → agentic_analyze → END
#     sub "agentic_explore"           → agentic_explore → END
#
# Follow-up prompts (is_followup=True) are meant to be sent as a SECOND TURN
# inside the same active agentic session (after the matching initial prompt
# completes and the frontend shows "Continuing session" mode). They demonstrate
# the back-and-forth interactability of the agentic pathways.
#
# Each entry is a plain dict so the FastAPI backend can serialise it directly
# to JSON via Pydantic without any extra conversion step.
# ---------------------------------------------------------------------------

DEMO_PROMPTS: list[dict] = [
    # ------------------------------------------------------------------
    # Route 1 — retrieve
    # Path: classify_task → retrieve_data → END
    # No analysis runs. The model fetches rows and returns them verbatim.
    # ------------------------------------------------------------------
    {
        "id": "retrieve_brake_complaints",
        "route": "retrieve",
        "route_label": "Retrieve",
        "label": "Browse Brake Failure Complaints",
        "description": (
            "Pure data retrieval — no analysis. Fetches 5 recent brake-failure "
            "complaints and returns the complaint text, make, model, year, and "
            "injury count. Exercises the classify_task → retrieve_data → END path."
        ),
        "prompt_text": (
            "Return the 5 most recent brake failure complaints from the database. "
            "For each record, include the complaint ID, make, model, model year, injury count, "
            "and the full complaint description text. "
            "No analysis, scoring, or interpretation needed — just return the raw records."
        ),
        "is_followup": False,
        "followup_for": None,
    },

    # ------------------------------------------------------------------
    # Route 2 — analyze
    # Path: classify_task → retrieve_data → analyze → confirm_csv_write → END
    # retrieve_data fetches the rows first; analyze applies a fixed per-row
    # analysis prompt (Safety_Prompt or Specific_Subsystem_Prompt) chosen by
    # gpt-5.1; confirm_csv_write asks whether to persist results to disk.
    # ------------------------------------------------------------------
    {
        "id": "analyze_safety_airbag",
        "route": "analyze",
        "route_label": "Analyze",
        "label": "Safety Evaluation — Airbag Malfunction Complaints",
        "description": (
            "Retrieve-then-analyze pipeline. Fetches 3 recent airbag complaints "
            "and runs a structured safety evaluation on each: danger level, urgency "
            "for human review, safety category, and full reasoning. "
            "Exercises the classify_task → retrieve_data → analyze → confirm_csv_write path."
        ),
        "prompt_text": (
            "Evaluate the 3 most recent airbag malfunction complaints for safety risk. "
            "For each complaint, assign a danger level (None / Low / Mild / Moderate / High / Severe), "
            "an urgency rating for human review (Low / Urgent / Emergent), a safety category, "
            "and detailed reasoning that explains the assessment. "
            "Apply the same structured safety evaluation to each of the 3 complaints."
        ),
        "is_followup": False,
        "followup_for": None,
    },

    # ------------------------------------------------------------------
    # Route 3a — agentic_analyze (initial turn)
    # Path: classify_task → classify_agentic_subtype → agentic_analyze → END
    # The agent retrieves targeted rows, optionally pulls supporting columns,
    # and emits a final structured JSON verdict delivered via TTS.
    # ------------------------------------------------------------------
    {
        "id": "agentic_analyze_toyota_brakes",
        "route": "agentic_analyze",
        "route_label": "Agentic Analyze",
        "label": "Deep-Dive Toyota Brake Complaints — Danger Rating (Initial)",
        "description": (
            "Initial turn of an agentic analysis session. The agent retrieves the "
            "5 most recent Toyota brake complaints, pulls injury and death counts as "
            "supporting evidence, and rates the danger level of each complaint with "
            "specific, evidence-backed reasoning. "
            "Exercises classify_task → classify_agentic_subtype → agentic_analyze."
        ),
        "prompt_text": (
            "Investigate the 5 most recent Toyota brake-related complaints from 2022 onward. "
            "For each complaint you retrieve, look up its injury count and death count from "
            "the database as supporting evidence. Then, using both the complaint text and the "
            "casualty data you pulled, produce a structured danger assessment for each complaint: "
            "rate it Low / Moderate / High / Severe and explain the specific evidence — from the "
            "complaint text and the injury and death figures — that justifies each rating."
        ),
        "is_followup": False,
        "followup_for": None,
    },

    # ------------------------------------------------------------------
    # Route 3b — agentic_analyze (follow-up turn)
    # Send AFTER "agentic_analyze_toyota_brakes" completes in the same session.
    # Demonstrates the back-and-forth of the agentic_analyze path by asking
    # the agent to revisit its verdicts in light of population-level data.
    # ------------------------------------------------------------------
    {
        "id": "agentic_analyze_toyota_brakes_followup",
        "route": "agentic_analyze",
        "route_label": "Agentic Analyze",
        "label": "Cross-Reference Population Data — Revise Ratings (Follow-up)",
        "description": (
            "Second turn — send this after the Toyota brake analysis completes "
            "and the session is in 'Continuing session' mode. Asks the agent to "
            "compare its per-complaint danger ratings against the full population "
            "of Toyota brake complaints and revise any ratings the broader data "
            "warrants changing. Demonstrates agentic back-and-forth reasoning."
        ),
        "prompt_text": (
            "Now retrieve the full population of Toyota brake complaints in the database "
            "and calculate the average injury count across all of them. Compare that population "
            "baseline against the 5 complaints you already assessed — do the broader numbers "
            "suggest any of your danger ratings are too high or too low? Revise any ratings "
            "the population data warrants changing and explain exactly what shifted and why."
        ),
        "is_followup": True,
        "followup_for": "agentic_analyze_toyota_brakes",
    },

    # ------------------------------------------------------------------
    # Route 4a — agentic_explore (initial turn)
    # Path: classify_task → classify_agentic_subtype → agentic_explore → END
    # The agent browses data conversationally, renders charts on request,
    # and narrates findings via TTS (user cannot see chart images directly).
    # ------------------------------------------------------------------
    {
        "id": "agentic_explore_top_makes_chart",
        "route": "agentic_explore",
        "route_label": "Agentic Explore",
        "label": "Chart: Top 10 Makes by Complaint Volume (Initial)",
        "description": (
            "Initial turn of an agentic exploration session. The agent renders a "
            "bar chart of the top 10 vehicle makes by total complaint count, then "
            "narrates the distribution — which make dominates, by how much, and any "
            "surprising entries. Exercises classify_task → classify_agentic_subtype "
            "→ agentic_explore with chart generation and spoken narration."
        ),
        "prompt_text": (
            "Let's explore the NHTSA complaint database by vehicle make. Generate a bar chart "
            "of the top 10 vehicle makes by total complaint count. Once the chart is rendered, "
            "walk me through what you see: which make leads, how large is the gap between first "
            "and second, and whether anything in the top 10 is surprising. "
            "I may want to drill deeper from there."
        ),
        "is_followup": False,
        "followup_for": None,
    },

    # ------------------------------------------------------------------
    # Route 4b — agentic_explore (follow-up turn)
    # Send AFTER "agentic_explore_top_makes_chart" completes in the same session.
    # Demonstrates multi-turn conversational chart exploration by drilling into
    # the leading make from the first chart.
    # ------------------------------------------------------------------
    {
        "id": "agentic_explore_top_makes_followup",
        "route": "agentic_explore",
        "route_label": "Agentic Explore",
        "label": "Drill Into Top Make — Subsystem Breakdown Chart (Follow-up)",
        "description": (
            "Second turn — send this after the top-makes chart completes and the "
            "session is in 'Continuing session' mode. Asks the agent to zoom into "
            "the #1 make and render a second chart showing its complaints broken down "
            "by subsystem component. Demonstrates chained chart generation and "
            "multi-turn conversational data exploration."
        ),
        "prompt_text": (
            "Let's keep exploring. Zoom into the make with the most complaints and break down "
            "its complaints by vehicle component or subsystem. Generate a chart showing which "
            "systems fail most often for that brand, then walk me through the top 5 failure "
            "categories and what they tell us about that make's reliability profile."
        ),
        "is_followup": True,
        "followup_for": "agentic_explore_top_makes_chart",
    },
]
