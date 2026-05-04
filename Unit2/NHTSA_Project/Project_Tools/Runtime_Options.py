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


def set_audio_enabled(enabled: bool) -> None:
    """Persist the current process-wide audio mode for downstream tools."""
    global _AUDIO_ENABLED
    _AUDIO_ENABLED = bool(enabled)


def is_audio_enabled() -> bool:
    """Return True when this process should use STT/TTS instead of text I/O."""
    return _AUDIO_ENABLED
