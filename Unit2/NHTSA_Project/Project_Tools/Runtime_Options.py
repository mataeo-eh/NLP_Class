"""
Runtime_Options.py
------------------
Process-wide runtime flags shared by the LangGraph entrypoint and tool modules.

The LangGraph CLI decides whether audio interaction is enabled for the current
process. Tool modules consult that decision so every user-facing interaction
uses one consistent mode:

- audio enabled  -> microphone capture + TTS playback
- audio disabled -> plain stdin/stdout text interaction

The defaults are intentionally conservative: unless the entrypoint explicitly
opts in, tools assume text mode so debugging never accidentally pulls in the
optional audio stack.
"""

_AUDIO_ENABLED = False

# Voice-model selector — chosen once at process start by the LangGraph CLI and
# read by Audio_Playback.generate_TTS_audio on every TTS call. Three values are
# supported:
#
#   "kokoro"   — local MLX Kokoro-82M-bf16 model (default). Renders sentence by
#                sentence into the AudioPlayer queue via mlx_audio.generate_audio,
#                which is what the rest of the pipeline has historically used.
#   "cartesia" — Cartesia cloud TTS (sonic-3.5). Cartesia natively streams PCM
#                bytes back over HTTP, so the per-sentence batching pipeline is
#                bypassed entirely: the full transcript is sent in one POST and
#                bytes start playing through the speakers as soon as they arrive.
#   "deepgram" — Deepgram cloud TTS (aura-2). Like Cartesia, Deepgram begins
#                emitting bytes as soon as it has audio to send, so the full
#                transcript is sent in one call and chunks are streamed
#                straight to the speakers — no Kokoro-style batching.
#
# The string is stored lowercased so callers never need to worry about
# capitalization (the CLI accepts "Cartesia"/"cartesia"/"CARTESIA"/etc.).
_VOICE_MODEL = "kokoro"
_VALID_VOICE_MODELS = ("kokoro", "cartesia", "deepgram")


def set_audio_enabled(enabled: bool) -> None:
    """Persist the current process-wide audio mode for downstream tools."""
    global _AUDIO_ENABLED
    _AUDIO_ENABLED = bool(enabled)


def is_audio_enabled() -> bool:
    """Return True when this process should use STT/TTS instead of text I/O."""
    return _AUDIO_ENABLED


def set_voice_model(model: str) -> None:
    """
    Persist the process-wide TTS engine choice.

    Accepts any capitalization of "kokoro" or "cartesia" — the value is
    normalised to lowercase before being stored. Raises ValueError on any other
    input so a typo at the CLI surfaces immediately rather than silently
    falling back to the default model.
    """
    global _VOICE_MODEL
    if not isinstance(model, str):
        raise ValueError(
            f"set_voice_model expected a string, got {type(model).__name__}."
        )
    normalized = model.strip().lower()
    if normalized not in _VALID_VOICE_MODELS:
        raise ValueError(
            f"Unknown voice model {model!r}. Expected one of "
            f"{', '.join(_VALID_VOICE_MODELS)} (case-insensitive)."
        )
    _VOICE_MODEL = normalized


def get_voice_model() -> str:
    """Return the currently selected voice model: "kokoro" or "cartesia"."""
    return _VOICE_MODEL
