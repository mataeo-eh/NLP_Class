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

    # Sub-type of the agentic task, set by classify_agentic_subtype (Nodes.py)
    # after classify_task has already confirmed task_type ==
    # "agentic_retrieve_and_analyze". Allowed values:
    #   - "agentic_analyze"  — the user has a specific question; the agent should
    #                          retrieve targeted data and produce a structured
    #                          analysis (mirrors the non-agentic analyze path).
    #   - "agentic_explore"  — the user wants open-ended exploration; the agent
    #                          should browse broadly, surface patterns, and
    #                          summarise findings without a fixed output schema.
    #   - ""                 — default / not-yet-classified; only valid before
    #                          classify_agentic_subtype has run.
    # Populated by: classify_agentic_subtype (Nodes.py)
    # Consumed by:  route_agentic_subtype (Edges.py) to fan out to the correct
    #               agentic sub-graph node.
    agentic_subtype: str

    # Accumulator of per-iteration metadata for the agentic loop. Each entry is
    # appended by the agentic loop node at the end of one reasoning iteration.
    # Shape of each entry:
    #   {
    #       "iter":       int,        # 1-based iteration counter
    #       "model":      str,        # model ID string used in that iteration
    #       "tool_calls": list[str],  # names of tools invoked in that iteration
    #       "summary":    str,        # one-sentence narrative of what happened
    #   }
    # Populated by: agentic_loop (Nodes.py) — one append per loop iteration.
    # Consumed by:  (a) debugging — inspect the log to understand what the agent
    #               did at each step; (b) TTS summary node — flatten the log into
    #               a human-readable narration of the agent's reasoning chain.
    iteration_log: list[dict]