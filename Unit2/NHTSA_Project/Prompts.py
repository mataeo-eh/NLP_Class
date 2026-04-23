import pandas as pd
from pathlib import Path
import inspect
from pprint import pprint

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


pprint(_load_nhtsa_taxonomy())