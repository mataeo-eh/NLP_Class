from State import State


def route_by_task_type(state: State) -> str:
    # Both "analyze" and "retrieve" enter retrieve_data first — data must be
    # fetched before the analyze node can reason over it. Only the agentic path
    # bypasses retrieve_data and manages its own retrieval loop internally.
    task_type = state["task_type"]

    if task_type in ("analyze", "retrieve"):
        return "retrieve_data"
    elif task_type == "agentic_retrieve_and_analyze":
        return "agentic_loop"


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