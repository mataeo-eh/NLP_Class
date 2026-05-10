import argparse
import json
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from Nodes import (
    classify_task,
    classify_agentic_subtype,  # NEW: classifies agentic sub-intent after task_type is set
    retrieve_data,
    analyze,
    confirm_csv_write,         # NEW: user yes/no gate before any CSV mutation
    csv_append,
    agentic_analyze,           # NEW: targeted agentic analysis path
    agentic_explore,           # NEW: open-ended agentic exploration path
)
from State import State
from Edges import (
    route_by_task_type,
    route_after_retrieve,
    route_csv_confirmation,    # NEW: routes confirm_csv_write to csv_append or END
    route_agentic_subtype,
)
from config import mercury_llm
from Project_Tools.Runtime_Options import (
    set_audio_enabled,
    set_voice_model,
    set_voice_preset,
    get_default_voice_preset,
    get_allowed_voice_presets,
)


'''
To run with CLI flags run the command below
python Unit2/NHTSA_Project/LangGraph/Graph.py -a -v deepgram
or
python Unit2/NHTSA_Project/LangGraph/Graph.py -a -v cartesia
or
python Unit2/NHTSA_Project/LangGraph/Graph.py -a -v kokoro

If --voice-model / -v is omitted, the pipeline defaults to "deepgram". The
selected key is just a short selector — the concrete TTS model path / id for
each engine is owned by Project_Tools/Audio_Playback.py and is never passed in
by callers, so no step of the pipeline can pin itself to a specific TTS model.

A voice preset can be selected with --voice-preset / -vp. The acceptable
preset values DEPEND on the chosen --voice-model: each engine exposes a
different concept of "voice" (kokoro voice name, cartesia voice UUID,
deepgram model id), so presets are not interchangeable. The full registry
lives in Project_Tools/Runtime_Options._ALLOWED_VOICE_PRESETS. If -vp is
omitted, the per-engine default is used (af_sky for kokoro,
6ccbfb76-1fc6-48f7-b71d-91ac6298247b for cartesia, aura-2-hyperion-en for
deepgram). Passing a preset that is not in the allowlist for the chosen
engine causes the CLI to exit with a clear error.

Example: python Unit2/NHTSA_Project/LangGraph/Graph.py -a -v deepgram -vp aura-2-hyperion-en
'''

WELCOME_TTS_TEXT = "Welcome back! What NHTSA adventure shall we embark on?"


PIPELINE_STATE_KEYS = {
    "user_request",
    "task_type",
    "query_result",
    "reasoning_output",
    "analysis",
    "analysis_prompt_name",
    "response",
    "csv_write_confirmed",
    "agentic_subtype",
    "conversation_messages",
    "resume_agentic_session",
    "iteration_log",
}


def _extract_final_tts_payload(data: dict | str) -> dict | str:
    """
    Reduce a LangGraph final-state envelope to the user-facing payload for TTS.

    LangGraph's compiled app.invoke(...) returns the full updated state dict.
    Only state["response"] should reach Mercury for the final spoken summary;
    the rest of the state is internal pipeline bookkeeping.
    """
    parsed: dict | str = data
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            return data

    if isinstance(parsed, dict) and "response" in parsed:
        state_keys_present = PIPELINE_STATE_KEYS.intersection(parsed.keys())
        if len(parsed) == 1 or len(state_keys_present) > 1:
            return parsed["response"]

    return parsed


def json_to_spoken_text(data: dict | str) -> str:
    """
    Convert the final pipeline payload into natural spoken prose for TTS.

    The caller may pass the full LangGraph final state, the already-extracted
    response field, or a JSON string produced by an upstream node. In all
    cases, normalize to the user-facing response payload first.
    """
    data = _extract_final_tts_payload(data)

    # If the payload is already plain prose, return it directly. If it is a
    # JSON string (for example the agentic analyze node's structured verdict),
    # parse it so Mercury can verbalize the structured content cleanly.
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
    # --voice-model / -v selects which TTS engine drives every spoken response
    # in this run. `type=str.lower` is argparse's documented way to normalise
    # the user's input before validation, so "Deepgram", "deepgram", and
    # "DEEPGRAM" all map to the same canonical value. `choices=` then enforces
    # the closed set — any other spelling produces an immediate parser error.
    #
    # The CLI value is just a short selector key — the concrete model path /
    # model id for each engine lives inside Project_Tools/Audio_Playback.py
    # (one default constant per engine: _DEFAULT_MODEL_PATH for kokoro,
    # _CARTESIA_DEFAULT_MODEL for cartesia, _DEEPGRAM_DEFAULT_MODEL for
    # deepgram). No call site in the pipeline passes a TTS model path itself;
    # they all rely on get_voice_model() dispatch in generate_TTS_audio.
    parser.add_argument(
        "-v",
        "--voice-model",
        type=str.lower,
        choices=("kokoro", "cartesia", "deepgram"),
        default="deepgram",
        help=(
            "Which TTS engine to use for spoken output (case-insensitive). "
            "Default is 'deepgram'. "
            "'deepgram' uses the Deepgram aura-2 cloud TTS, which streams "
            "audio chunks directly to the speakers as they are synthesised "
            "(requires DEEPGRAM_TTS_KEY). "
            "'cartesia' uses the Cartesia sonic-3.5 cloud TTS, which natively "
            "streams audio bytes directly to the speakers — the per-sentence "
            "batching pipeline is bypassed and the full transcript is sent in "
            "one request (requires TTS_KEY). "
            "'kokoro' uses the local MLX Kokoro-82M-bf16 model with the "
            "per-sentence batching pipeline."
        ),
    )
    # --voice-preset / -vp selects which voice preset of the chosen engine to
    # use. argparse cannot express "the allowed choices for this flag depend on
    # the value of another flag," so we accept any string here and validate
    # against the per-engine allowlist post-parse (see _resolve_voice_preset
    # below). Default is None — meaning "use the default preset for the
    # selected engine," which is the same value that's currently wired in.
    #
    # Each engine exposes a different concept of "voice":
    #   kokoro   — voice name (e.g. "af_sky") passed to mlx_audio.generate_audio.
    #   cartesia — voice UUID passed as `voice.id` in the JSON body.
    #   deepgram — model id (e.g. "aura-2-hyperion-en") that bakes voice
    #              and language together.
    # Presets are NOT interchangeable across engines — the registry in
    # Runtime_Options enforces this with a per-engine allowlist.
    parser.add_argument(
        "-vp",
        "--voice-preset",
        type=str,
        default=None,
        help=(
            "Voice preset to use with the selected --voice-model. The set of "
            "allowed values depends on -v: kokoro accepts voice names "
            "(e.g. 'af_sky'), cartesia accepts voice UUIDs, and deepgram "
            "accepts model ids that bake voice and language together "
            "(e.g. 'aura-2-hyperion-en'). When omitted, the default preset "
            "for the selected engine is used. Currently only one preset is "
            "wired in per engine; passing anything else raises a CLI error."
        ),
    )
    return parser.parse_args()


def _resolve_voice_preset(model: str, preset: str | None) -> str:
    """
    Pick the final voice preset to use for this run.

    `model` is the already-validated CLI --voice-model value. `preset` is the
    raw --voice-preset argument (or None if the user omitted the flag). When
    None, fall back to the per-engine default. When non-None, validate against
    the engine's allowlist and exit with a clean error on mismatch — argparse
    cannot do this validation natively because the allowed set depends on
    another flag's value.
    """
    if preset is None:
        return get_default_voice_preset(model)

    allowed = get_allowed_voice_presets(model)
    if preset not in allowed:
        # Mirror argparse's own error UX: write to stderr and exit non-zero so
        # CI / shell wrappers see a clear failure rather than a Python
        # traceback.
        import sys as _sys
        _sys.stderr.write(
            f"error: voice preset {preset!r} is not allowed for "
            f"--voice-model {model!r}. Allowed presets: "
            f"{', '.join(allowed)}.\n"
        )
        raise SystemExit(2)
    return preset


graph = StateGraph(State)

graph.add_node("classify_task", classify_task)
# classify_agentic_subtype further categorises an agentic request into either
# "agentic_analyze" (targeted question) or "agentic_explore" (open-ended browse).
# It runs only when classify_task has already confirmed the agentic task_type.
graph.add_node("classify_agentic_subtype", classify_agentic_subtype)
graph.add_node("retrieve_data", retrieve_data)
graph.add_node("analyze", analyze)
# confirm_csv_write speaks a Mercury-summary of the pending write via TTS, then
# asks the user yes/no through voice_ask_user (or stdin in text mode). Its boolean
# output drives the conditional edge below — this is what makes the CSV write
# optional so the same graph runs locally (where writes are allowed) and in any
# hosted environment where the agent must not touch the filesystem.
graph.add_node("confirm_csv_write", confirm_csv_write)
graph.add_node("csv_append", csv_append)
# agentic_analyze: the user posed a specific question; agent retrieves targeted
# data, reasons over it, and returns a structured analysis — mirrors the
# non-agentic analyze path but with full tool-calling autonomy.
graph.add_node("agentic_analyze", agentic_analyze)
# agentic_explore: the user wants open-ended exploration; agent browses broadly,
# surfaces patterns, and summarises findings without a fixed output schema.
graph.add_node("agentic_explore", agentic_explore)

graph.add_edge(START, "classify_task")   # entry point
# After analysis, hand off to the user-confirmation gate instead of writing
# directly to disk. confirm_csv_write narrates a Mercury+TTS summary of what's
# queued, asks the user yes/no, and stores the answer in state so the
# conditional edge below can decide whether to actually call csv_append.
graph.add_edge("analyze", "confirm_csv_write")
# csv_append handles dedup against existing df_index values, so re-running
# analysis on a previously analyzed complaint never produces a duplicate row —
# but it only runs when the user has explicitly approved the write via the
# confirm_csv_write gate above.
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

# After confirm_csv_write asks the user yes/no, route based on the recorded answer.
# This is the gate that decouples analysis from filesystem writes: when the user
# says yes, the graph proceeds into csv_append exactly like before; when the user
# says no (or the reply is ambiguous, or there's nothing to write), the graph
# exits without touching disk. That makes the same compiled graph safe to run
# both locally (writes allowed) and in any hosted setting (writes forbidden).
graph.add_conditional_edges(
    "confirm_csv_write",
    route_csv_confirmation,
    {
        "csv_append": "csv_append",
        "END":        END,
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
    # Publish the voice-model selector before any TTS call happens. argparse
    # has already normalised the value to lowercase ("deepgram", "cartesia",
    # or "kokoro") via type=str.lower + choices, so set_voice_model receives a
    # valid input. Doing this even when audio_available is False is harmless —
    # the selector is only read inside generate_TTS_audio, which never runs in
    # text mode — and it keeps the runtime state consistent with what the user
    # requested in case audio is re-enabled later in the process.
    set_voice_model(args.voice_model)
    # Resolve and publish the voice preset for the chosen engine. Order matters:
    # set_voice_model() must run FIRST because set_voice_preset() validates the
    # supplied preset against the allowlist for the currently selected model.
    # _resolve_voice_preset substitutes the per-engine default when -vp was
    # omitted and exits with a clean CLI error when the user passed a preset
    # that is not allowed for the chosen engine.
    resolved_preset = _resolve_voice_preset(args.voice_model, args.voice_preset)
    set_voice_preset(resolved_preset)
    if audio_available:
        print(
            f"[Graph] Using voice model: {args.voice_model} "
            f"(preset: {resolved_preset})"
        )

    # Greet the user via TTS so they know the system is ready, then capture their
    # spoken request and transcribe it before handing off to the graph.
    #
    # Neither `model=` nor `voice=` is passed here — generate_TTS_audio reads
    # both from the runtime selectors set above (set_voice_model and
    # set_voice_preset) and dispatches to the right engine internally. The
    # kokoro-shaped lang_code / streaming_interval / stream arguments are
    # only read by the kokoro branch and ignored by cartesia / deepgram,
    # which encode language in their model id and stream natively.
    if audio_available:
        generate_TTS_audio(
            text=WELCOME_TTS_TEXT,
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
        "csv_write_confirmed": False, # bool — set by confirm_csv_write based on user yes/no
        "agentic_subtype": "",       # str — set by classify_agentic_subtype; empty until then
        "iteration_log": [],         # list[dict] — appended by agentic nodes each iteration
    })

    # LangGraph returns the full final state here. The final Mercury+TTS
    # handoff should speak only state["response"], not the rest of the
    # bookkeeping fields, so json_to_spoken_text normalizes that envelope first.
    spoken_response = json_to_spoken_text(result)
    if audio_available:
        # Same dispatch contract as the greeting call above — neither `model=`
        # nor `voice=` is passed. generate_TTS_audio reads both from the
        # runtime selectors and routes to the engine the user picked on the CLI.
        generate_TTS_audio(
            text=spoken_response,
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
