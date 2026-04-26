from typing_extensions import TypedDict


class State(TypedDict):
    user_request: str          # transcribed STT input
    task_type: str             # "analyze", "retrieve", "agentic_retrieve_and_analyze"

    # Structured rows produced by retrieve_data. Each dict is one row from the
    # parquet DB or a CSV output file, and always carries an "index" field that
    # holds the row's original position in the source. Shape depends on task_type:
    #   - "analyze":  rows are projected to {"index": int, "CDESCR": str} so the
    #                 analyze node only sees the complaint text it is supposed
    #                 to reason over (per the prompt design in Prompts.py).
    #   - "retrieve": rows carry every non-NaN column the retrieval tool returned,
    #                 since the user is consuming the data directly via TTS.
    query_result: list[dict]

    reasoning_output: str      # intermediate thinking, exposed to user via TTS

    # Per-row analysis output produced by the analyze node. One entry per row in
    # query_result. Each entry is {"index": int, "result": <parsed JSON dict>}
    # where "result" is whatever JSON shape the chosen analysis prompt produced
    # (e.g. Safety_Prompt yields safety_category / level_of_danger / urgency /
    # reasoning; Specific_Subsystem_Prompt yields subsystems / confidence / reasoning).
    analysis: list[dict]

    # Name of the prompt the analyze node selected from ANALYSIS_PROMPTS (e.g.
    # "Safety_Prompt"). Set explicitly so downstream nodes — csv_append in
    # particular — can route results to the correct per-prompt CSV file without
    # having to parse it back out of reasoning_output prose.
    analysis_prompt_name: str

    response: str              # final text, ready to be sent to TTS