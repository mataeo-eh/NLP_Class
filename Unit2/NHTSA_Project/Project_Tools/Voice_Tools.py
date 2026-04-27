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
  - Audio_Playback loads the Kokoro-82M-bf16 MLX TTS model on first import.
  - Audio_Capture loads the Whisper STT model on first import.
The first call to this module will therefore be slow (model load). All
subsequent calls within the same process are fast — the models stay resident
in memory.

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

# ---------------------------------------------------------------------------
# Path wiring — Audio_Playback and Audio_Capture live in the same directory
# as this file (Project_Tools/), so the package root is one level up.
# sys.path.insert ensures the package is importable even when this module is
# executed from a working directory other than the project root.
# This matches the import pattern used in NHTSA_Query_Tools.py.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from Project_Tools.Audio_Playback import generate_TTS_audio
from Project_Tools.Audio_Capture import capture_and_transcribe


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
    try:
        # -- Step 1: TTS playback ------------------------------------------
        # Synthesise and play the question using the Metal-backed Kokoro model.
        # Parameters match the canonical voice config established in Graph.py's
        # __main__ block and replicated in NHTSA_Query_Tools.Ask_User:
        #   model            — MLX Kokoro-82M-bf16, Apple Silicon / Metal-backed
        #   voice            — "af_sky": the project-standard female voice
        #   speed=0.85       — slightly slower than default for clarity
        #   lang_code="a"    — American English phoneme set
        #   play=True        — stream directly to AudioPlayer (no file write)
        #   streaming_interval=0.2 — queue each sentence chunk every 0.2 s
        #   stream=True      — yield chunks incrementally; suppresses audio_write()
        #   save=False       — never persist audio to disk
        generate_TTS_audio(
            text=question,
            model="mlx-community/Kokoro-82M-bf16",
            voice="af_sky",
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
