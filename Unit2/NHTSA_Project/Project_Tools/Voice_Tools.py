"""
Voice_Tools.py
--------------
Canonical voice wrapper for agentic LangGraph nodes.

Role
----
Provides a single LangChain @tool — voice_ask_user — that speaks a question
to the user via TTS, then captures and returns their spoken reply as text.
Agentic nodes call this tool when they need to clarify ambiguity, ask a
follow-up question, or confirm before taking a destructive action.

Model caching
-------------
Both underlying audio modules (Audio_Playback and Audio_Capture) load their
ML models at import time:
  - Audio_Playback pre-loads the local Kokoro-82M-bf16 MLX TTS model so the
    "kokoro" CLI selector has no cold start. The "cartesia" and "deepgram"
    selectors hit cloud APIs and have no local model to load.
  - Audio_Capture loads the Whisper STT model on first import.
The first call to this module will therefore be slow (model load). All
subsequent calls within the same process are fast — the models stay resident
in memory.

TTS engine selection
--------------------
voice_ask_user does NOT pick a TTS model. The engine is chosen process-wide
by the LangGraph CLI via --voice-model (default: deepgram) and read inside
Audio_Playback.generate_TTS_audio. This module simply calls generate_TTS_audio
without a model argument so every spoken interaction in the run uses the same
engine the user selected.

Blocking behaviour
------------------
voice_ask_user is BLOCKING: it waits for TTS playback to finish, then waits
for the user to speak and VAD to detect end-of-speech, then waits for Whisper
transcription to complete. The agentic node must not call this tool on the
main event loop. Parallelism is provided by the calling node's
ThreadPoolExecutor (standard LangGraph tool-execution pattern).
"""

import sys
from pathlib import Path
from langchain_core.tools import tool
from Project_Tools.Runtime_Options import is_audio_enabled, is_headless_mode

# ---------------------------------------------------------------------------
# Path wiring — Audio_Playback and Audio_Capture live in the same directory
# as this file (Project_Tools/), so the package root is one level up.
# sys.path.insert ensures the package is importable even when this module is
# executed from a working directory other than the project root.
# This matches the import pattern used in NHTSA_Query_Tools.py.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))


@tool
def voice_ask_user(question: str) -> str:
    """
    Speak `question` to the user via TTS, then listen for and transcribe their
    spoken response. Returns the transcribed text.

    Use this tool to:
      - Clarify ambiguity in the user's request that would cause a wrong or
        unhelpful result if left unresolved.
      - Ask a follow-up question when additional context is needed.
      - Confirm intent before taking a destructive or irreversible action.

    Parameters
    ----------
    question : str
        The question to speak aloud to the user.

    Returns
    -------
    str
        The Whisper transcript of the user's spoken reply, or an error string
        starting with '[voice tool error:' if audio I/O fails.
    """
    # Headless mode (FastAPI on Render): there is no human at stdin and no
    # microphone / speaker on the server, so the tool must NOT call input() or
    # try to capture audio — both would either block forever or crash. Instead
    # we return a structured refusal string. The agentic loop's prompts
    # already instruct the model to "adapt rather than retry" when a tool
    # signals it cannot complete, so the LLM will keep going with whatever
    # information it already has.
    if is_headless_mode():
        return (
            "[voice_ask_user unavailable: this process is running in a hosted "
            "headless environment with no microphone, speakers, or stdin. "
            "Proceed with the data you already have and produce your best "
            "answer without asking the user a follow-up question.]"
        )

    if not is_audio_enabled():
        # Keep the tool contract identical in text mode: the LLM still calls the
        # same tool name, but the interaction shifts to plain terminal I/O so the
        # graph can run without the optional speech stack.
        print(f"[voice_ask_user] {question}")
        return input("Your response: ").strip()

    try:
        from Project_Tools.Audio_Playback import generate_TTS_audio
        from Project_Tools.Audio_Capture import capture_and_transcribe

        # -- Step 1: TTS playback ------------------------------------------
        # Synthesise and play the question using whichever TTS engine + voice
        # preset the LangGraph CLI selected for this run. The engine (deepgram
        # / cartesia / kokoro) and preset are read inside generate_TTS_audio
        # via Runtime_Options.get_voice_model() and get_voice_preset(); the
        # concrete model path / id for that engine lives entirely inside
        # Audio_Playback.py — this call site never names a TTS model OR a
        # voice directly.
        #
        # Parameters match the canonical voice config established in Graph.py's
        # __main__ block and replicated in NHTSA_Query_Tools.Ask_User:
        #   speed=0.85       — slightly slower than default for clarity
        #   lang_code="a"    — American English phoneme set (kokoro path only)
        #   play=True        — stream directly to AudioPlayer (no file write)
        #   streaming_interval=0.2 — queue each sentence chunk every 0.2 s (kokoro path)
        #   stream=True      — yield chunks incrementally; suppresses audio_write() (kokoro path)
        #   save=False       — never persist audio to disk
        # The kokoro-only kwargs are silently ignored on the cartesia and
        # deepgram branches; voice/language for those engines are baked into
        # the runtime voice preset (cartesia voice UUID, deepgram model id).
        generate_TTS_audio(
            text=question,
            speed=0.85,
            lang_code="a",
            play=True,
            streaming_interval=0.2,
            stream=True,
            save=False,
        )

        # -- Step 2: STT capture ------------------------------------------
        # Open the microphone, record until VAD detects end-of-speech, run
        # Whisper transcription, and return the transcript string.
        # capture_and_transcribe() takes no arguments — all recording
        # parameters (VAD threshold, silence duration, sample rate) are
        # configured as module-level constants in Audio_Capture.py.
        transcript = capture_and_transcribe()

        return transcript

    except Exception as exc:
        # Return a structured error string rather than re-raising.
        # The agentic loop must not crash on voice failure — the LLM can
        # inspect this string and fall back to text-based clarification.
        return f"[voice tool error: {exc}]"
