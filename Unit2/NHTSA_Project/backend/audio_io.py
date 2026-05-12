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
from collections.abc import Iterator
from email.message import Message
from io import BytesIO
from pathlib import Path

import requests
from tts_registry import (
    get_default_hosted_voice_preset,
    resolve_hosted_tts_selection,
)


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
_OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"
_OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
_OPENAI_STREAM_SAMPLE_RATE = 24000
_OPENAI_STREAM_CHANNELS = 1
_OPENAI_STREAM_PCM_ENCODING = "pcm16"

# The hosted backend still reuses shared project modules above backend/ (for
# example pipeline_runner imports Graph -> Project_Tools.*). This small path
# bootstrap lets audio_io.py import those shared modules when needed and also
# keeps direct local smoke tests working from the backend/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Deepgram Speak defaults already used by Project_Tools.Audio_Playback:
#   model       -> Deepgram Aura voice/model id (voice is baked into the id)
#   encoding    -> linear16 raw little-endian PCM
#   sample_rate -> 24000 Hz for Aura 2 voices
#
# The hosted frontend uses the same raw PCM stream for low-latency playback in
# the browser's Web Audio API, then wraps the accumulated bytes into WAV only
# when it needs a replayable/downloadable asset after streaming finishes.
_DEEPGRAM_SAMPLE_RATE = 24000
_DEEPGRAM_CHANNELS = 1
_DEEPGRAM_PCM_ENCODING = "linear16"
_CARTESIA_TTS_URL = "https://api.cartesia.ai/tts/bytes"
_CARTESIA_VERSION = os.environ.get("CARTESIA_VERSION", "2025-04-16")
_CARTESIA_TTS_MODEL = os.environ.get("CARTESIA_TTS_MODEL", "sonic-3.5")
_CARTESIA_SAMPLE_RATE = 44100
_CARTESIA_CHANNELS = 1
_CARTESIA_PCM_ENCODING = "pcm_f32le"
_HTTP_STREAM_CHUNK_SIZE = 4096


def _safe_filename(filename: str | None) -> str:
    """Return a provider-friendly filename for the multipart upload."""
    if not filename:
        return "audio.webm"
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix in _ALLOWED_AUDIO_SUFFIXES:
        return safe_name
    return "audio.webm"


def _normalized_content_type(content_type: str | None) -> str | None:
    """
    Return the base media type from the browser-supplied Content-Type header.

    Browser recorders are allowed to append RFC-style parameters such as
    `codecs=mp4a.40.2`. FastAPI surfaces that full header string through
    UploadFile.content_type, so we parse it with the standard library's MIME
    header handling instead of comparing the raw string directly.
    """
    if not content_type:
        return None

    message = Message()
    message["Content-Type"] = content_type
    normalized = message.get_content_type()
    if normalized == "text/plain":
        return content_type.strip().lower()
    return normalized


def validate_audio_upload(content_type: str | None, payload: bytes) -> None:
    """
    Validate the uploaded audio envelope before forwarding it to OpenAI.

    FastAPI's UploadFile gives us the browser-provided filename, content type,
    and bytes. We use those concrete fields directly:

    * payload length enforces empty-upload and max-size rules.
    * content_type is normalized to its base media type before validation, so
      browser codec parameters like `audio/mp4; codecs=mp4a.40.2` remain valid.
    * filename is handled separately as a provider hint and is never trusted for
      security decisions.
    """
    if not payload:
        raise ValueError("Uploaded audio file is empty.")
    if len(payload) > MAX_AUDIO_UPLOAD_BYTES:
        raise ValueError(
            f"Uploaded audio is {len(payload)} bytes; limit is {MAX_AUDIO_UPLOAD_BYTES} bytes."
        )
    normalized_content_type = _normalized_content_type(content_type)
    if normalized_content_type and normalized_content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
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


def _cartesia_headers() -> dict[str, str]:
    """Build Cartesia auth headers from Railway's CARTESIA_API_KEY secret."""
    api_key = os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        raise RuntimeError("Cartesia TTS is not configured. Set CARTESIA_API_KEY in Railway.")
    return {
        "Authorization": f"Bearer {api_key}",
        "Cartesia-Version": _CARTESIA_VERSION,
        "Content-Type": "application/json",
    }


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


def _deepgram_model_id(voice_preset: str | None = None) -> str:
    """
    Return the Deepgram voice/model id for one hosted speech request.

    Deepgram bakes voice and language into one model id (for example
    aura-2-hyperion-en). The hosted frontend can send an explicit voice preset
    per request. When it omits one, the backend preserves the historical
    default path: DEEPGRAM_TTS_MODEL env override first, otherwise the hosted
    registry default.
    """
    if voice_preset:
        return voice_preset
    env_model = os.environ.get("DEEPGRAM_TTS_MODEL")
    if env_model:
        return env_model

    return get_default_hosted_voice_preset("deepgram")


def _deepgram_api_key() -> str:
    """Return the Deepgram TTS key using the existing local env-var name first."""
    api_key = os.environ.get("DEEPGRAM_TTS_KEY") or os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Deepgram TTS is not configured. Set DEEPGRAM_TTS_KEY or "
            "DEEPGRAM_API_KEY in Railway."
        )
    return api_key


def _synthesize_openai_speech_wav(
    *,
    text: str,
    voice_preset: str,
) -> bytes:
    """
    Synthesize WAV bytes through OpenAI's hosted speech endpoint.

    Documented request fields handled here:
    * model           -> gpt-4o-mini-tts by default, configurable via env
    * input           -> raw text to speak
    * voice           -> one built-in OpenAI voice id selected by the frontend
    * response_format -> wav so the browser can play the response directly
    """
    response = requests.post(
        _OPENAI_TTS_URL,
        headers=_openai_headers(json_body=True),
        json={
            "model": _OPENAI_TTS_MODEL,
            "input": text,
            "voice": voice_preset,
            "response_format": "wav",
        },
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenAI speech request failed: HTTP {response.status_code}: {response.text}"
        )
    if not response.content:
        raise RuntimeError("OpenAI speech request returned no audio bytes.")
    return response.content


def _stream_openai_speech_pcm(
    *,
    text: str,
    voice_preset: str,
) -> Iterator[bytes]:
    """
    Stream OpenAI speech as raw PCM chunks for the hosted backend.

    The hosted route explicitly requests `response_format="pcm"` and
    `stream_format="audio"` from `/v1/audio/speech`, then forwards the
    incremental response body to the browser unchanged.

    Assumed contract for the frontend:
    * codec       -> pcm16
    * sample_rate -> 24000
    * channels    -> 1
    """
    response = requests.post(
        _OPENAI_TTS_URL,
        headers=_openai_headers(json_body=True),
        json={
            "model": _OPENAI_TTS_MODEL,
            "input": text,
            "voice": voice_preset,
            "response_format": "pcm",
            "stream_format": "audio",
        },
        stream=True,
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"OpenAI speech stream request failed: HTTP {response.status_code}: "
            f"{response.text}"
        )
    return _iter_http_audio_chunks(response)


def _synthesize_cartesia_speech_wav(
    *,
    text: str,
    voice_preset: str,
) -> bytes:
    """
    Synthesize WAV bytes through Cartesia's bytes endpoint.

    Documented request fields handled here:
    * model_id       -> Sonic model id, configurable via env
    * transcript     -> raw text to speak
    * voice.mode/id  -> hosted registry voice id chosen by the frontend
    * output_format  -> explicit WAV container so the browser can replay it
    """
    response = requests.post(
        _CARTESIA_TTS_URL,
        headers=_cartesia_headers(),
        json={
            "model_id": _CARTESIA_TTS_MODEL,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": voice_preset,
            },
            "output_format": {
                "container": "wav",
                "encoding": "pcm_f32le",
                "sample_rate": _CARTESIA_SAMPLE_RATE,
            },
        },
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Cartesia speech request failed: HTTP {response.status_code}: {response.text}"
        )
    if not response.content:
        raise RuntimeError("Cartesia speech request returned no audio bytes.")
    return response.content


def _iter_http_audio_chunks(response: requests.Response) -> Iterator[bytes]:
    """
    Normalize one HTTP streaming response into non-empty byte chunks.

    Cartesia's bytes endpoint streams the response body incrementally. We keep
    the contract explicit here: only non-empty `bytes` chunks are yielded, and
    the underlying response is always closed when iteration ends or aborts.
    """
    try:
        saw_audio = False
        for chunk in response.iter_content(chunk_size=_HTTP_STREAM_CHUNK_SIZE):
            if not chunk:
                continue
            saw_audio = True
            yield chunk
        if not saw_audio:
            raise RuntimeError("The streaming TTS provider returned no audio bytes.")
    finally:
        response.close()


def _iter_deepgram_audio_chunks(response) -> Iterator[bytes]:
    """
    Normalize documented Deepgram SDK response variants into a byte iterator.

    Deepgram SDK examples show two concrete shapes:
    * an object with `response.stream.getvalue()` for buffered output
    * an iterator of `bytes` chunks for streaming output

    The backend handles both documented variants explicitly and rejects anything
    else instead of guessing.
    """
    if isinstance(response, bytes):
        yield response
        return
    if isinstance(response, bytearray):
        yield bytes(response)
        return

    stream = getattr(response, "stream", None)
    if stream is not None and hasattr(stream, "getvalue"):
        buffered = stream.getvalue()
        if not buffered:
            raise RuntimeError("Deepgram TTS returned no audio bytes.")
        yield buffered
        return

    try:
        saw_audio = False
        for chunk in response:
            if not chunk:
                continue
            if isinstance(chunk, bytes):
                saw_audio = True
                yield chunk
                continue
            if isinstance(chunk, bytearray):
                saw_audio = True
                yield bytes(chunk)
                continue
            raise RuntimeError(
                "Deepgram TTS yielded a non-bytes chunk; expected bytes or bytearray."
            )
        if not saw_audio:
            raise RuntimeError("Deepgram TTS returned no audio bytes.")
    except TypeError as exc:
        raise RuntimeError(
            "Deepgram TTS returned an unsupported response shape; expected "
            "bytes, a stream buffer, or an iterator of byte chunks."
        ) from exc


def _deepgram_chunks_to_bytes(response) -> bytes:
    """
    Collect a documented Deepgram streaming/buffered response into one byte string.

    This is used by the buffered WAV route, while the streaming route forwards
    the same chunks directly to the browser.
    """
    return b"".join(_iter_deepgram_audio_chunks(response))


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
    tts_provider: str | None = None,
    voice_preset: str | None = None,
) -> bytes:
    """
    Synthesize text as browser-playable WAV bytes through the selected provider.

    The hosted frontend sends provider/voice choices on every request instead
    of mutating one shared process-wide selector. That keeps concurrent browser
    sessions isolated and lets the frontend switch providers mid-run: each new
    narration or final-answer speech request can choose a different route.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Text is required for speech synthesis.")

    resolved = resolve_hosted_tts_selection(tts_provider, voice_preset)
    resolved_provider = resolved["tts_provider"]
    resolved_voice = resolved["voice_preset"]

    if resolved_provider == "openai":
        return _synthesize_openai_speech_wav(
            text=cleaned_text,
            voice_preset=resolved_voice,
        )

    if resolved_provider == "cartesia":
        return _synthesize_cartesia_speech_wav(
            text=cleaned_text,
            voice_preset=resolved_voice,
        )

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
        model=_deepgram_model_id(resolved_voice),
        encoding=_DEEPGRAM_PCM_ENCODING,
        sample_rate=_DEEPGRAM_SAMPLE_RATE,
    )
    pcm_audio = _deepgram_chunks_to_bytes(response)
    return _pcm_s16le_to_wav(pcm_audio, _DEEPGRAM_SAMPLE_RATE)


def synthesize_speech_pcm_stream(
    text: str,
    *,
    tts_provider: str | None = None,
    voice_preset: str | None = None,
) -> Iterator[bytes]:
    """
    Stream raw audio chunks for browser playback from one supported hosted TTS provider.

    Request/response contract
    -------------------------
    The selected provider determines the raw byte format:

    * Deepgram -> `linear16`, 24000 Hz, mono
    * Cartesia -> `pcm_f32le`, 44100 Hz, mono

    The frontend reads the explicit HTTP headers emitted by FastAPI and uses
    the Web Audio API to schedule playback chunk-by-chunk after a user gesture
    unlocks the audio context.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Text is required for speech synthesis.")

    resolved = resolve_hosted_tts_selection(tts_provider, voice_preset)
    resolved_provider = resolved["tts_provider"]
    resolved_voice = resolved["voice_preset"]

    if resolved_provider == "openai":
        return _stream_openai_speech_pcm(
            text=cleaned_text,
            voice_preset=resolved_voice,
        )

    if resolved_provider == "cartesia":
        response = requests.post(
            _CARTESIA_TTS_URL,
            headers=_cartesia_headers(),
            json={
                "model_id": _CARTESIA_TTS_MODEL,
                "transcript": cleaned_text,
                "voice": {
                    "mode": "id",
                    "id": resolved_voice,
                },
                "output_format": {
                    "container": "raw",
                    "encoding": _CARTESIA_PCM_ENCODING,
                    "sample_rate": _CARTESIA_SAMPLE_RATE,
                },
            },
            stream=True,
            timeout=120,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Cartesia speech stream request failed: HTTP {response.status_code}: "
                f"{response.text}"
            )
        return _iter_http_audio_chunks(response)

    if resolved_provider != "deepgram":
        raise ValueError(
            f"Hosted PCM streaming is not available for provider "
            f"{resolved_provider!r}. Use /audio/speech instead."
        )

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
        model=_deepgram_model_id(resolved_voice),
        encoding=_DEEPGRAM_PCM_ENCODING,
        sample_rate=_DEEPGRAM_SAMPLE_RATE,
    )
    return _iter_deepgram_audio_chunks(response)


def get_streaming_audio_contract(
    *,
    tts_provider: str | None = None,
    voice_preset: str | None = None,
) -> dict[str, int | str]:
    """
    Return the exact raw-audio contract for one streaming speech request.

    This mirrors synthesize_speech_pcm_stream() so the FastAPI route can emit
    deterministic headers before the first chunk leaves the server.
    """
    resolved = resolve_hosted_tts_selection(tts_provider, voice_preset)
    resolved_provider = resolved["tts_provider"]
    if resolved_provider == "openai":
        return {
            "tts_provider": resolved_provider,
            "codec": _OPENAI_STREAM_PCM_ENCODING,
            "sample_rate": _OPENAI_STREAM_SAMPLE_RATE,
            "channels": _OPENAI_STREAM_CHANNELS,
        }
    if resolved_provider == "deepgram":
        return {
            "tts_provider": resolved_provider,
            "codec": _DEEPGRAM_PCM_ENCODING,
            "sample_rate": _DEEPGRAM_SAMPLE_RATE,
            "channels": _DEEPGRAM_CHANNELS,
        }
    if resolved_provider == "cartesia":
        return {
            "tts_provider": resolved_provider,
            "codec": _CARTESIA_PCM_ENCODING,
            "sample_rate": _CARTESIA_SAMPLE_RATE,
            "channels": _CARTESIA_CHANNELS,
        }
    raise ValueError(
        f"Hosted PCM streaming is not available for provider {resolved_provider!r}."
    )
