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

    # User confirmation gate between the analyze node and csv_append. The
    # confirm_csv_write node sets this flag to True when the user explicitly
    # approves writing analysis results to disk, and False otherwise (including
    # ambiguous replies, no rows to write, or the user saying "no"). The
    # conditional edge route_csv_confirmation reads this flag to decide whether
    # to route into csv_append or skip straight to END.
    #
    # Why this exists:
    #   The pipeline must run safely in two very different deployment modes:
    #     - Local interactive mode: the agent has filesystem write access and the
    #       user wants analysis results persisted to the per-prompt CSVs.
    #     - Hosted / read-only mode: the agent is forbidden from writing files,
    #       so the user always answers "no" and csv_append is bypassed.
    #   Routing on this boolean keeps the same graph definition usable in both
    #   modes — the only thing that changes is the user's spoken answer.
    #
    # Populated by: confirm_csv_write (Nodes.py).
    # Consumed by:  route_csv_confirmation (Edges.py).
    csv_write_confirmed: bool

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