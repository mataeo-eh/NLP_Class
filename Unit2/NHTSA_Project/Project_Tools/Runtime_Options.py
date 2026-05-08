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
# read by Audio_Playback.generate_TTS_audio on every TTS call. The selector is a
# short, user-facing key; the concrete model path / model id for each engine is
# encapsulated inside Audio_Playback.py (one default constant per engine), so
# nothing outside this module needs to know what the actual TTS model is.
#
# Three values are supported:
#
#   "deepgram" — Deepgram cloud TTS (aura-2, default). Deepgram begins emitting
#                bytes as soon as it has audio to send, so the full transcript
#                is sent in one call and chunks are streamed straight to the
#                speakers — no Kokoro-style per-sentence batching.
#   "cartesia" — Cartesia cloud TTS (sonic-3.5). Cartesia natively streams PCM
#                bytes back over HTTP, so the per-sentence batching pipeline is
#                bypassed entirely: the full transcript is sent in one POST and
#                bytes start playing through the speakers as soon as they arrive.
#   "kokoro"   — local MLX Kokoro-82M-bf16 model. Renders sentence by sentence
#                into the AudioPlayer queue via mlx_audio.generate_audio.
#
# The string is stored lowercased so callers never need to worry about
# capitalization (the CLI accepts "Deepgram"/"deepgram"/"DEEPGRAM"/etc.).
#
# Default is "deepgram" — when the user does not pass --voice-model on the CLI
# the pipeline falls back to Deepgram rather than the local Kokoro model.
_VOICE_MODEL = "deepgram"
_VALID_VOICE_MODELS = ("kokoro", "cartesia", "deepgram")


# Voice-preset registry — per-engine. Each TTS engine exposes a different
# concept of "voice":
#
#   kokoro   — voice name string passed to mlx_audio.generate_audio(voice=...).
#              Example: "af_sky" (American female, default).
#   cartesia — voice UUID passed as `voice.id` in the Cartesia tts/bytes JSON
#              body. Example: "6ccbfb76-1fc6-48f7-b71d-91ac6298247b".
#   deepgram — Deepgram bakes voice and language into its model id, so the
#              "preset" for deepgram is literally the model id passed to
#              client.speak.v1.audio.generate(model=...). Example:
#              "aura-2-hyperion-en".
#
# Because the wire format differs per engine, presets are NOT interchangeable
# across engines — a kokoro voice name will not work with cartesia, etc. The
# registry below is the single source of truth: the CLI validates -vp against
# the entry for the selected -v engine, and Audio_Playback's per-engine branch
# routes the resolved preset to the correct underlying parameter.
#
# For now each engine has only one allowed preset (its current default). New
# presets will be added here as they are wired in.
_ALLOWED_VOICE_PRESETS = {
    "kokoro":   ("af_sky",),
    "cartesia": ("6ccbfb76-1fc6-48f7-b71d-91ac6298247b",),
    "deepgram": ("aura-2-hyperion-en",),
}

# Per-engine default — used when the user does not pass --voice-preset on the
# CLI. Always a member of the engine's _ALLOWED_VOICE_PRESETS tuple above.
_DEFAULT_VOICE_PRESET = {
    "kokoro":   "af_sky",
    "cartesia": "6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
    "deepgram": "aura-2-hyperion-en",
}

# Explicit user-selected preset, or None if the user did not pass --voice-preset
# (in which case get_voice_preset() falls back to _DEFAULT_VOICE_PRESET for the
# currently selected voice model).
_VOICE_PRESET: str | None = None


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

    Accepts any capitalization of "deepgram", "cartesia", or "kokoro" — the
    value is normalised to lowercase before being stored. Raises ValueError on
    any other input so a typo at the CLI surfaces immediately rather than
    silently falling back to the default model.
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
    """Return the currently selected voice model: "deepgram", "cartesia", or "kokoro"."""
    return _VOICE_MODEL


def get_allowed_voice_presets(model: str) -> tuple[str, ...]:
    """
    Return the tuple of voice presets allowed for the given engine.

    Raises ValueError for unknown engines so the CLI can surface typos
    immediately rather than silently accepting an arbitrary preset string.
    """
    normalized = model.strip().lower()
    if normalized not in _ALLOWED_VOICE_PRESETS:
        raise ValueError(
            f"Unknown voice model {model!r}. Expected one of "
            f"{', '.join(_VALID_VOICE_MODELS)} (case-insensitive)."
        )
    return _ALLOWED_VOICE_PRESETS[normalized]


def get_default_voice_preset(model: str) -> str:
    """
    Return the default voice preset for the given engine.

    Used by the CLI to fill in --voice-preset when the flag is omitted, and
    by get_voice_preset() as the runtime fallback when no explicit preset
    has been set for the process.
    """
    normalized = model.strip().lower()
    if normalized not in _DEFAULT_VOICE_PRESET:
        raise ValueError(
            f"Unknown voice model {model!r}. Expected one of "
            f"{', '.join(_VALID_VOICE_MODELS)} (case-insensitive)."
        )
    return _DEFAULT_VOICE_PRESET[normalized]


def set_voice_preset(preset: str) -> None:
    """
    Persist the process-wide voice-preset choice.

    The preset is validated against _ALLOWED_VOICE_PRESETS for the currently
    selected voice model — call set_voice_model() FIRST so this function can
    check membership against the right engine's allowlist. Raises ValueError
    when the preset is not allowed for the current engine, so a CLI typo
    surfaces immediately instead of silently being accepted by an engine that
    does not understand it.
    """
    global _VOICE_PRESET
    if not isinstance(preset, str):
        raise ValueError(
            f"set_voice_preset expected a string, got {type(preset).__name__}."
        )
    allowed = _ALLOWED_VOICE_PRESETS[_VOICE_MODEL]
    if preset not in allowed:
        raise ValueError(
            f"Voice preset {preset!r} is not allowed for voice model "
            f"{_VOICE_MODEL!r}. Allowed presets: {', '.join(allowed)}."
        )
    _VOICE_PRESET = preset


def get_voice_preset() -> str:
    """
    Return the active voice preset for the currently selected voice model.

    Falls back to the per-engine default when set_voice_preset() has not been
    called for the process — this keeps Audio_Playback's per-engine branches
    working even when the module is imported outside the LangGraph CLI (e.g.
    from a notebook). The returned value is always a member of
    _ALLOWED_VOICE_PRESETS[get_voice_model()].
    """
    if _VOICE_PRESET is not None:
        return _VOICE_PRESET
    return _DEFAULT_VOICE_PRESET[_VOICE_MODEL]
