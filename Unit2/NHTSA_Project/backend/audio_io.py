"""
Backend audio I/O helpers.

This module is intentionally separate from Project_Tools.Audio_Capture and
Project_Tools.Audio_Playback. Those modules are built for the local CLI: they
open a microphone, play through local speakers, and import hardware-oriented
packages such as sounddevice. A hosted FastAPI backend has a different contract:

* Input audio arrives as uploaded bytes from the browser.
* Output audio must be returned as HTTP response bytes the browser can play.
* Speech work should be delegated to hosted audio APIs so Railway does not load
  Whisper, Silero VAD, torch, ffmpeg, Kokoro, or microphone/speaker packages.

The public functions below keep that web contract explicit and deterministic.
"""

from __future__ import annotations

import os
import sys
import wave
from io import BytesIO
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Upload and synthesis limits.
# ---------------------------------------------------------------------------
# MAX_AUDIO_UPLOAD_BYTES mirrors OpenAI's Audio API upload limit. Keeping the
# same boundary means the backend rejects impossible requests before forwarding
# bytes over the network.
MAX_AUDIO_UPLOAD_BYTES = int(os.environ.get("MAX_AUDIO_UPLOAD_BYTES", str(25 * 1024 * 1024)))

# Browsers commonly send webm/ogg/wav/mp4/mpeg. application/octet-stream is
# allowed because some MediaRecorder/browser combinations omit a specific codec
# media type even though the file extension remains useful to the API provider.
ALLOWED_AUDIO_CONTENT_TYPES = {
    "application/octet-stream",
    "audio/aac",
    "audio/flac",
    "audio/mpga",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/webm",
    "audio/x-wav",
    "video/mp4",
    "video/webm",
}

_ALLOWED_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mpeg",
    ".mp3",
    ".mp4",
    ".mpga",
    ".oga",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}


# ---------------------------------------------------------------------------
# OpenAI audio API configuration.
# ---------------------------------------------------------------------------
# API contract from OpenAI Audio transcriptions:
#   POST /v1/audio/transcriptions as multipart/form-data
#   file             -> uploaded audio file object
#   model            -> gpt-4o-mini-transcribe, gpt-4o-transcribe, etc.
#   language         -> optional ISO-639-1 code, improves accuracy/latency
#   response_format  -> json for the 4o transcription models
#   chunking_strategy -> "auto" asks OpenAI to use server-side VAD boundaries
# Response JSON includes at least `text`. We return text plus request metadata
# and intentionally avoid exposing provider internals as frontend contract.
_OPENAI_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
_OPENAI_STT_MODEL = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")

# Keep backend TTS aligned with the existing CLI voice configuration. The
# Project_Tools modules live one directory above backend/, so this small path
# bootstrap lets audio_io.py be imported both through main.py and directly in
# local smoke tests.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Deepgram Speak defaults already used by Project_Tools.Audio_Playback:
#   model       -> Deepgram Aura voice/model id (voice is baked into the id)
#   encoding    -> linear16 raw little-endian PCM
#   sample_rate -> 24000 Hz for Aura 2 voices
_DEEPGRAM_SAMPLE_RATE = 24000


def _safe_filename(filename: str | None) -> str:
    """Return a provider-friendly filename for the multipart upload."""
    if not filename:
        return "audio.webm"
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix in _ALLOWED_AUDIO_SUFFIXES:
        return safe_name
    return "audio.webm"


def validate_audio_upload(content_type: str | None, payload: bytes) -> None:
    """
    Validate the uploaded audio envelope before forwarding it to OpenAI.

    FastAPI's UploadFile gives us the browser-provided filename, content type,
    and bytes. We use those concrete fields directly:

    * payload length enforces empty-upload and max-size rules.
    * content_type must be one of the accepted browser audio/video media types.
    * filename is handled separately as a provider hint and is never trusted for
      security decisions.
    """
    if not payload:
        raise ValueError("Uploaded audio file is empty.")
    if len(payload) > MAX_AUDIO_UPLOAD_BYTES:
        raise ValueError(
            f"Uploaded audio is {len(payload)} bytes; limit is {MAX_AUDIO_UPLOAD_BYTES} bytes."
        )
    if content_type and content_type.lower() not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise ValueError(f"Unsupported audio content type: {content_type}")


def _openai_headers(*, json_body: bool = False) -> dict[str, str]:
    """Build OpenAI auth headers from Railway's OPENAI_API_KEY secret."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI audio API is not configured. Set OPENAI_API_KEY in Railway.")
    headers = {"Authorization": f"Bearer {api_key}"}
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def transcribe_audio_bytes(
    payload: bytes,
    filename: str | None,
    content_type: str | None,
    language: str = "en",
) -> dict:
    """
    Transcribe uploaded audio bytes through OpenAI's hosted STT API.

    `chunking_strategy="auto"` delegates voice activity boundary detection to
    the provider. That replaces local Silero VAD for this hosted workflow while
    still letting the frontend send one browser-recorded blob.
    """
    validate_audio_upload(content_type=content_type, payload=payload)

    file_name = _safe_filename(filename)
    media_type = content_type or "application/octet-stream"
    response = requests.post(
        _OPENAI_TRANSCRIPTIONS_URL,
        headers=_openai_headers(),
        data={
            "model": _OPENAI_STT_MODEL,
            "language": language,
            "response_format": "json",
            "chunking_strategy": "auto",
        },
        files={"file": (file_name, payload, media_type)},
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenAI transcription request failed: HTTP {response.status_code}: {response.text}"
        )

    result = response.json()
    return {
        "text": (result.get("text") or "").strip(),
        "language": result.get("language") or language,
        "model": _OPENAI_STT_MODEL,
        "provider": "openai",
        "chunking_strategy": "auto",
    }


def _deepgram_model_id() -> str:
    """
    Return the existing project-selected Deepgram voice/model id.

    Runtime_Options is the project's source of truth for voice presets. For
    Deepgram, the "preset" is the model id itself, for example
    aura-2-hyperion-en. We use the same default here so the hosted backend voice
    matches the local CLI unless Railway explicitly sets DEEPGRAM_TTS_MODEL.
    """
    env_model = os.environ.get("DEEPGRAM_TTS_MODEL")
    if env_model:
        return env_model

    from Project_Tools.Runtime_Options import get_default_voice_preset  # noqa: PLC0415

    return get_default_voice_preset("deepgram")


def _deepgram_api_key() -> str:
    """Return the Deepgram TTS key using the existing local env-var name first."""
    api_key = os.environ.get("DEEPGRAM_TTS_KEY") or os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Deepgram TTS is not configured. Set DEEPGRAM_TTS_KEY or "
            "DEEPGRAM_API_KEY in Railway."
        )
    return api_key


def _deepgram_chunks_to_bytes(response) -> bytes:
    """
    Normalize documented Deepgram SDK response variants into one byte string.

    Deepgram SDK examples show two concrete shapes:
    * an object with `response.stream.getvalue()` for buffered output
    * an iterator of `bytes` chunks for streaming output

    The backend handles both documented variants explicitly and rejects anything
    else instead of guessing.
    """
    if isinstance(response, bytes):
        return response
    if isinstance(response, bytearray):
        return bytes(response)
    stream = getattr(response, "stream", None)
    if stream is not None and hasattr(stream, "getvalue"):
        return stream.getvalue()

    try:
        return b"".join(chunk for chunk in response if chunk)
    except TypeError as exc:
        raise RuntimeError(
            "Deepgram TTS returned an unsupported response shape; expected "
            "bytes, a stream buffer, or an iterator of byte chunks."
        ) from exc


def _pcm_s16le_to_wav(pcm_audio: bytes, sample_rate: int) -> bytes:
    """Wrap Deepgram's raw mono linear16 PCM bytes in a browser-playable WAV container."""
    if not pcm_audio:
        raise RuntimeError("Deepgram TTS returned no audio bytes.")

    output = BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_audio)
    return output.getvalue()


def synthesize_speech_wav(
    text: str,
    *,
    speed: float = 1.0,
) -> bytes:
    """
    Synthesize text as WAV bytes through Deepgram's hosted TTS API.

    Deepgram's Speak v1 SDK accepts text, model, encoding, sample_rate, and
    speed. The existing project preset supplies the model id, so callers do not
    send a voice/model override from the frontend. We request raw linear16 PCM
    to match the local Deepgram path, then wrap it in WAV for browser playback.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Text is required for speech synthesis.")

    try:
        from deepgram import DeepgramClient  # noqa: PLC0415
    except Exception as exc:
        raise RuntimeError(
            "Deepgram TTS requires the `deepgram-sdk` package, which could not "
            f"be imported. Original error: {exc}"
        ) from exc

    client = DeepgramClient(api_key=_deepgram_api_key())
    response = client.speak.v1.audio.generate(
        text=cleaned_text,
        model=_deepgram_model_id(),
        encoding="linear16",
        sample_rate=_DEEPGRAM_SAMPLE_RATE,
        speed=float(speed),
    )
    pcm_audio = _deepgram_chunks_to_bytes(response)
    return _pcm_s16le_to_wav(pcm_audio, _DEEPGRAM_SAMPLE_RATE)
