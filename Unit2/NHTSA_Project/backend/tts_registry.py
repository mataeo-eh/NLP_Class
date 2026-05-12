"""
Hosted TTS provider and voice registry.

This module is intentionally backend-specific. The local LangGraph CLI already
has its own process-wide runtime selectors in Project_Tools.Runtime_Options,
but the hosted web flow has a different contract:

* The browser needs a deterministic list of providers and voices so it can
  render provider/voice dropdowns without guessing.
* The hosted backend must not depend on a single shared process-wide voice
  selection because different browser tabs can choose different voices.
* Each speech request should therefore carry the provider + voice it wants to
  use, and this module validates those values against one explicit registry.

The defaults preserve the current hosted behaviour:
* provider -> Deepgram
* voice    -> aura-2-hyperion-en
"""

from __future__ import annotations

from copy import deepcopy


DEFAULT_HOSTED_TTS_PROVIDER = "deepgram"
DEFAULT_HOSTED_TTS_PREVIEW_TEXT = (
    "Hello, how are you doing on this fine day. "
    "Were you able to find everything you were looking for?"
)


_HOSTED_TTS_PROVIDERS = {
    "deepgram": {
        "id": "deepgram",
        "label": "Deepgram",
        "description": (
            "Deepgram Aura voices. This is the current hosted default and the "
            "only provider wired into the low-latency PCM streaming route."
        ),
        "supports_streaming": True,
        "default_voice_preset": "aura-2-hyperion-en",
        "voices": [
            {
                "id": "aura-2-hyperion-en",
                "label": "Hyperion",
                "description": "Australian English, warm and empathetic. Current default.",
            },
            {
                "id": "aura-2-thalia-en",
                "label": "Thalia",
                "description": "American English, clear and energetic.",
            },
            {
                "id": "aura-2-andromeda-en",
                "label": "Andromeda",
                "description": "American English, casual and expressive.",
            },
            {
                "id": "aura-2-helena-en",
                "label": "Helena",
                "description": "American English, caring and natural.",
            },
            {
                "id": "aura-2-asteria-en",
                "label": "Asteria",
                "description": "American English, confident and knowledgeable.",
            },
            {
                "id": "aura-2-athena-en",
                "label": "Athena",
                "description": "American English, calm and professional.",
            },
            {
                "id": "aura-2-apollo-en",
                "label": "Apollo",
                "description": "American English, confident and casual.",
            },
            {
                "id": "aura-2-arcas-en",
                "label": "Arcas",
                "description": "American English, smooth and clear.",
            },
        ],
    },
    "openai": {
        "id": "openai",
        "label": "OpenAI",
        "description": (
            "OpenAI hosted TTS via /v1/audio/speech. The hosted backend uses "
            "the PCM streaming route for low-latency browser playback."
        ),
        "supports_streaming": True,
        "default_voice_preset": "alloy",
        "voices": [
            {"id": "alloy", "label": "Alloy", "description": "Built-in OpenAI voice."},
            {"id": "ash", "label": "Ash", "description": "Built-in OpenAI voice."},
            {"id": "ballad", "label": "Ballad", "description": "Built-in OpenAI voice."},
            {"id": "coral", "label": "Coral", "description": "Built-in OpenAI voice."},
            {"id": "echo", "label": "Echo", "description": "Built-in OpenAI voice."},
            {"id": "fable", "label": "Fable", "description": "Built-in OpenAI voice."},
            {"id": "onyx", "label": "Onyx", "description": "Built-in OpenAI voice."},
            {"id": "nova", "label": "Nova", "description": "Built-in OpenAI voice."},
            {"id": "sage", "label": "Sage", "description": "Built-in OpenAI voice."},
            {"id": "shimmer", "label": "Shimmer", "description": "Built-in OpenAI voice."},
            {"id": "verse", "label": "Verse", "description": "Built-in OpenAI voice."},
            {"id": "marin", "label": "Marin", "description": "Built-in OpenAI voice."},
            {"id": "cedar", "label": "Cedar", "description": "Built-in OpenAI voice."},
        ],
    },
    "cartesia": {
        "id": "cartesia",
        "label": "Cartesia",
        "description": (
            "Cartesia Sonic voices via /tts/bytes. The backend currently uses "
            "its documented streamed-bytes path for low-latency hosted playback."
        ),
        "supports_streaming": True,
        "default_voice_preset": "6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
        "voices": [
            {
                "id": "6ccbfb76-1fc6-48f7-b71d-91ac6298247b",
                "label": "Tessa",
                "description": "American English, expressive. Current default.",
            },
            {
                "id": "f786b574-daa5-4673-aa0c-cbe3e8534c02",
                "label": "Katie",
                "description": "American English, stable voice for agents.",
            },
            {
                "id": "228fca29-3a0a-435c-8728-5cb483251068",
                "label": "Kiefer",
                "description": "American English, stable voice for agents.",
            },
            {
                "id": "c961b81c-a935-4c17-bfb3-ba2239de8c2f",
                "label": "Kyle",
                "description": "American English, expressive voice.",
            },
        ],
    },
}


def get_default_hosted_tts_provider() -> str:
    """Return the hosted provider used when the frontend sends no explicit choice."""
    return DEFAULT_HOSTED_TTS_PROVIDER


def get_default_hosted_tts_preview_text() -> str:
    """Return the default sample sentence shown in the voice-preview textarea."""
    return DEFAULT_HOSTED_TTS_PREVIEW_TEXT


def _normalized_provider(provider: str | None) -> str:
    """Normalize a provider id and validate it against the hosted registry."""
    candidate = (provider or DEFAULT_HOSTED_TTS_PROVIDER).strip().lower()
    if candidate not in _HOSTED_TTS_PROVIDERS:
        allowed = ", ".join(_HOSTED_TTS_PROVIDERS)
        raise ValueError(
            f"Unknown hosted TTS provider {provider!r}. Expected one of {allowed}."
        )
    return candidate


def get_hosted_tts_provider(provider: str | None) -> dict:
    """Return one provider definition from the hosted registry."""
    normalized = _normalized_provider(provider)
    return deepcopy(_HOSTED_TTS_PROVIDERS[normalized])


def list_hosted_tts_providers() -> list[dict]:
    """Return the full hosted provider list in a JSON-safe shape."""
    return [deepcopy(provider) for provider in _HOSTED_TTS_PROVIDERS.values()]


def get_default_hosted_voice_preset(provider: str | None) -> str:
    """Return the default voice preset for one hosted provider."""
    provider_info = get_hosted_tts_provider(provider)
    return str(provider_info["default_voice_preset"])


def resolve_hosted_tts_selection(
    tts_provider: str | None,
    voice_preset: str | None,
) -> dict[str, str]:
    """
    Resolve and validate one hosted provider + voice pair.

    The frontend may omit both values, in which case the current hosted default
    stays in force. When the provider is present but the voice is omitted, we
    fall back to that provider's default voice. This keeps the request contract
    small while still guaranteeing a deterministic voice selection.
    """
    provider_info = get_hosted_tts_provider(tts_provider)
    resolved_provider = str(provider_info["id"])
    allowed_voice_ids = {
        str(voice["id"]): voice for voice in provider_info.get("voices", [])
    }
    resolved_voice = (voice_preset or "").strip() or str(provider_info["default_voice_preset"])
    if resolved_voice not in allowed_voice_ids:
        allowed = ", ".join(allowed_voice_ids)
        raise ValueError(
            f"Voice preset {voice_preset!r} is not allowed for hosted TTS provider "
            f"{resolved_provider!r}. Allowed presets: {allowed}."
        )
    return {
        "tts_provider": resolved_provider,
        "voice_preset": resolved_voice,
    }


def hosted_provider_supports_streaming(tts_provider: str | None) -> bool:
    """Return whether the provider is wired into the hosted PCM streaming route."""
    provider_info = get_hosted_tts_provider(tts_provider)
    return bool(provider_info["supports_streaming"])
