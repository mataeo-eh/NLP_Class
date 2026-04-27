from State import State


def route_by_task_type(state: State) -> str:
    # Both "analyze" and "retrieve" enter retrieve_data first — data must be
    # fetched before the analyze node can reason over it. Only the agentic path
    # bypasses retrieve_data and manages its own retrieval loop internally.
    task_type = state["task_type"]

    if task_type in ("analyze", "retrieve"):
        return "retrieve_data"
    elif task_type == "agentic_retrieve_and_analyze":
        # Route into classify_agentic_subtype so the graph can fan out to
        # agentic_analyze or agentic_explore based on the user's intent.
        return "classify_agentic_subtype"


def route_after_retrieve(state: State) -> str:
    # After retrieve_data completes, decide whether to continue into analysis
    # or exit. Pure retrieval requests are done at this point; analyze requests
    # need to hand the fetched query_result off to the analyze node.
    task_type = state["task_type"]

    if task_type == "analyze":
        return "analyze"
    else:
        # "retrieve" (and any unexpected value) exits — nothing left to do
        return "END"


def route_agentic_subtype(state: State) -> str:
    # WHY this function exists:
    #   After classify_agentic_subtype runs, the graph needs to fan out to one
    #   of two agentic sub-nodes depending on what the classifier decided.
    #   This edge reads the already-set agentic_subtype and returns the
    #   matching node name so LangGraph can follow the correct branch.
    #
    # Deterministic fallback:
    #   If classify_agentic_subtype produced an unexpected string (e.g. the LLM
    #   hallucinated a label), we do not crash — we warn loudly and default to
    #   "agentic_explore". This mirrors the pattern used inside the analyze node
    #   when its prompt-name lookup fails: prefer a safe, observable default over
    #   a hard error mid-graph.
    #
    # Populated by: classify_agentic_subtype (Nodes.py)
    # Consumed by:  LangGraph conditional edge attached to classify_agentic_subtype
    _VALID = {"agentic_analyze", "agentic_explore"}
    subtype = state["agentic_subtype"]

    if subtype in _VALID:
        return subtype

    print(
        f"[route_agentic_subtype] unexpected value {subtype!r}; "
        f"defaulting to agentic_explore"
    )
    return "agentic_explore"