import argparse
import json
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from Nodes import (
    classify_task,
    classify_agentic_subtype,  # NEW: classifies agentic sub-intent after task_type is set
    retrieve_data,
    analyze,
    csv_append,
    agentic_analyze,           # NEW: targeted agentic analysis path
    agentic_explore,           # NEW: open-ended agentic exploration path
)
from State import State
from Edges import route_by_task_type, route_after_retrieve, route_agentic_subtype
from config import mercury_llm
from Project_Tools.Runtime_Options import set_audio_enabled


def json_to_spoken_text(data: dict | str) -> str:
    """
    Converts the pipeline's output (a State dict or JSON string) into natural
    spoken prose suitable for TTS. Mercury handles this well since it's a fast
    diffusion LLM — good fit for a lightweight formatting step at the end of
    the pipeline.
    """
    # If the data is already a plain string (e.g. response field was already prose),
    # try to parse it as JSON first; if that fails, return it as-is.
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data

    prompt = (
        "Convert the following data into clear, natural spoken language "
        "suitable for a text-to-speech system. No bullet points, no markdown, "
        "no JSON syntax — just flowing prose a listener can follow.\n\n"
        f"{json.dumps(data, indent=2)}"
    )

    response = mercury_llm.invoke([HumanMessage(content=prompt)])
    return response.content


def parse_cli_args() -> argparse.Namespace:
    """
    Parse CLI flags for the LangGraph entrypoint.

    Python's argparse documentation defines `action="store_true"` as the
    standard way to model an on/off flag: the parsed attribute is False when
    the flag is absent and True when the flag is present.
    """
    parser = argparse.ArgumentParser(
        description="Run the NHTSA LangGraph complaint-analysis pipeline.",
    )
    parser.add_argument(
        "-a",
        "--audio",
        action="store_true",
        help=(
            "Enable the existing speech pipeline so the run uses microphone "
            "input and spoken output instead of plain text I/O."
        ),
    )
    return parser.parse_args()


graph = StateGraph(State)

graph.add_node("classify_task", classify_task)
# classify_agentic_subtype further categorises an agentic request into either
# "agentic_analyze" (targeted question) or "agentic_explore" (open-ended browse).
# It runs only when classify_task has already confirmed the agentic task_type.
graph.add_node("classify_agentic_subtype", classify_agentic_subtype)
graph.add_node("retrieve_data", retrieve_data)
graph.add_node("analyze", analyze)
graph.add_node("csv_append", csv_append)
# agentic_analyze: the user posed a specific question; agent retrieves targeted
# data, reasons over it, and returns a structured analysis — mirrors the
# non-agentic analyze path but with full tool-calling autonomy.
graph.add_node("agentic_analyze", agentic_analyze)
# agentic_explore: the user wants open-ended exploration; agent browses broadly,
# surfaces patterns, and summarises findings without a fixed output schema.
graph.add_node("agentic_explore", agentic_explore)

graph.add_edge(START, "classify_task")   # entry point
# After analysis, persist each per-row result to the prompt's CSV before exit.
# csv_append handles dedup against existing df_index values, so re-running
# analysis on a previously analyzed complaint never produces a duplicate row.
graph.add_edge("analyze", "csv_append")
graph.add_edge("csv_append", END)
# Both agentic sub-nodes are terminal — they produce their own response and
# write iteration_log; no further graph nodes need to run after them.
graph.add_edge("agentic_analyze", END)
graph.add_edge("agentic_explore", END)

# classify_task routes "retrieve" and "analyze" both into retrieve_data,
# since data must be fetched before analysis can run.
# "agentic_retrieve_and_analyze" goes to classify_agentic_subtype first so the
# graph can fan out to the correct agentic sub-node based on user intent.
graph.add_conditional_edges(
    "classify_task",
    route_by_task_type,
    {
        "retrieve_data":           "retrieve_data",
        "classify_agentic_subtype": "classify_agentic_subtype",
    }
)

# After retrieve_data, pure retrieval tasks exit; analyze tasks continue onward.
graph.add_conditional_edges(
    "retrieve_data",
    route_after_retrieve,
    {
        "analyze": "analyze",
        "END":     END,
    }
)

# After classify_agentic_subtype sets state["agentic_subtype"], route_agentic_subtype
# reads that value and directs flow to the matching agentic execution node.
graph.add_conditional_edges(
    "classify_agentic_subtype",
    route_agentic_subtype,
    {
        "agentic_analyze": "agentic_analyze",
        "agentic_explore": "agentic_explore",
    }
)
app = graph.compile()


def main() -> None:
    """
    Run the LangGraph pipeline in either audio mode or text mode.

    The CLI flag is the single source of truth for whether the process should
    activate STT/TTS. If the user requested audio but optional audio imports
    fail, we immediately downgrade the shared runtime mode to text so every
    downstream tool follows the same fallback behavior.
    """
    args = parse_cli_args()
    requested_audio = args.audio
    audio_available = False
    generate_TTS_audio = None
    capture_and_transcribe = None

    if requested_audio:
        try:
            from Project_Tools.Audio_Playback import generate_TTS_audio
            from Project_Tools.Audio_Capture import capture_and_transcribe
            audio_available = True
        except Exception as exc:
            print(f"[Graph] Audio requested but unavailable; using text input/output. Reason: {exc}")
    else:
        print("[Graph] Audio disabled via CLI; using text input/output.")

    # Publish the effective mode once so every downstream node/tool uses the
    # same interaction style for this run.
    set_audio_enabled(audio_available)

    # Greet the user via TTS so they know the system is ready, then capture their
    # spoken request and transcribe it before handing off to the graph.
    if audio_available:
        generate_TTS_audio(
            text="Hey, welcome back. What are you looking for today?",
            model="mlx-community/Kokoro-82M-bf16",
            voice="af_sky",
            speed=0.85,
            lang_code="a",
            play=True,
            streaming_interval=0.2,
            stream=True,
            save=False,
        )
        user_request = capture_and_transcribe()
    else:
        user_request = input("What are you looking for today? ").strip()

    result = app.invoke({
        "user_request": user_request,
        "task_type": "",
        "query_result": [],          # list[dict] — populated by retrieve_data
        "reasoning_output": "",
        "analysis": [],              # list[dict] — populated by analyze (per-row results)
        "analysis_prompt_name": "",  # str — populated by analyze, consumed by csv_append
        "response": "",
        "agentic_subtype": "",       # str — set by classify_agentic_subtype; empty until then
        "iteration_log": [],         # list[dict] — appended by agentic nodes each iteration
    })

    spoken_response = json_to_spoken_text(result["response"])
    if audio_available:
        generate_TTS_audio(
            text=spoken_response,
            model="mlx-community/Kokoro-82M-bf16",
            voice="af_sky",
            speed=0.85,
            lang_code="a",
            play=True,
            streaming_interval=0.2,
            # stream=True prevents the library from writing audio to disk.
            # Without it, generate_audio always calls audio_write() regardless of play=True.
            # save defaults to False, so audio is only queued to AudioPlayer, never persisted.
            stream=True,
            save=False,
        )
    else:
        print(spoken_response)


if __name__ == "__main__":
    main()
