import os
import sys
import wave
import datetime as _dt
from dotenv import load_dotenv
from Project_Tools.Runtime_Options import is_audio_enabled, get_voice_model

load_dotenv()

# Pre-load the Kokoro model once at module import time so it sits in memory
# and is reused across all generate_TTS_audio calls. Loading takes several
# seconds; doing it here means the first TTS call has no cold-start penalty.
# lazy=False forces all weights into memory immediately rather than on first use.
_DEFAULT_MODEL_PATH = "mlx-community/Kokoro-82M-bf16"
_tts_model = None
_tts_import_error = None

try:
    from mlx_audio.tts.generate import generate_audio
    from mlx_audio.tts.utils import load_model
except Exception as exc:
    generate_audio = None
    load_model = None
    _tts_import_error = exc

# ---------------------------------------------------------------------------
# Cartesia configuration
# ---------------------------------------------------------------------------
# These constants mirror the values in the user's verified curl test against
# Cartesia's `tts/bytes` HTTP endpoint. We hit the same endpoint directly via
# `requests` rather than adding the cartesia SDK as a dependency — the endpoint
# returns audio bytes progressively, so streaming playback is achieved by
# piping `iter_content()` chunks straight into a sounddevice output stream.
#
# API contract reference (per the user's curl + Cartesia public docs):
#   POST https://api.cartesia.ai/tts/bytes
#   Headers:
#     Cartesia-Version: 2026-03-01
#     X-API-Key:        <TTS_KEY env var>
#     Content-Type:     application/json
#   Body fields used:
#     model_id           — "sonic-3.5" (latest as of the user's test)
#     transcript         — full text passed in one go (no per-sentence batching)
#     voice              — {"mode": "id", "id": <voice id>}
#     output_format      — {container, encoding, sample_rate}
#                          We use raw / pcm_s16le / 44100 so chunks can be fed
#                          directly to sounddevice.RawOutputStream without WAV
#                          header parsing or container demux.
#     language           — "en"
#     generation_config  — {"speed": <float>, "volume": 1}
# Response: streaming binary body of raw little-endian 16-bit mono PCM samples.
_CARTESIA_API_URL = "https://api.cartesia.ai/tts/bytes"
_CARTESIA_API_VERSION = "2026-03-01"
_CARTESIA_DEFAULT_MODEL = "sonic-3.5"
_CARTESIA_DEFAULT_VOICE_ID = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"
_CARTESIA_SAMPLE_RATE = 44100  # Hz — must match output_format.sample_rate below
_CARTESIA_HTTP_CHUNK = 4096    # bytes per requests.iter_content() chunk

# Lazy-imported optional deps for the Cartesia path. We do not import them at
# module load time so users running the existing Kokoro pipeline never need
# `requests` or `sounddevice` available — this preserves the original import
# behavior of this module.
_cartesia_import_error = None


# ---------------------------------------------------------------------------
# Deepgram configuration
# ---------------------------------------------------------------------------
# Deepgram's Speak v1 REST API exposes `client.speak.v1.audio.generate(...)`,
# which returns an Iterator[bytes] of the synthesised audio. We pick
# encoding="linear16" so the bytes are raw little-endian 16-bit PCM mono — the
# same wire format the Cartesia path uses — so the same sounddevice streaming
# path can play it without any container/codec handling.
#
# The Deepgram TTS pipeline begins emitting audio bytes as soon as the first
# phonemes are synthesised, so (like Cartesia) the full transcript is sent in
# one call and there is no need for the Kokoro per-sentence batching pipeline.
#
# API reference (Deepgram Python SDK v5):
#   client = DeepgramClient(api_key)
#   chunks = client.speak.v1.audio.generate(
#       text=...,
#       model="aura-2-hyperion-en",
#       encoding="linear16",
#       sample_rate=24000,
#   )
#   for chunk in chunks: ...   # raw PCM bytes, stream as they arrive
_DEEPGRAM_DEFAULT_MODEL = "aura-2-hyperion-en"
_DEEPGRAM_SAMPLE_RATE = 24000   # Hz — Deepgram's recommended rate for aura-2

# Lazy-imported optional dep for the Deepgram path. As with Cartesia, we do not
# import `deepgram` at module load time — only inside the dispatch function —
# so users on the Kokoro path never need the SDK installed.
_deepgram_import_error = None


def _get_tts_model():
    """Load the TTS model lazily so imports do not require the audio stack."""
    global _tts_model
    if _tts_import_error is not None:
        raise RuntimeError(
            "TTS audio is unavailable because mlx_audio could not be imported. "
            "Install optional audio dependencies on a supported platform to "
            f"enable spoken playback. Original error: {_tts_import_error}"
        )
    if _tts_model is None:
        _tts_model = load_model(model_path=_DEFAULT_MODEL_PATH, lazy=False)
    return _tts_model


def _generate_cartesia_audio(
    text: str,
    speed: float = 1.0,
    play: bool = True,
    save: bool = False,
    save_path: str | None = None,
    voice_id: str = _CARTESIA_DEFAULT_VOICE_ID,
    model_id: str = _CARTESIA_DEFAULT_MODEL,
) -> None:
    """
    Synthesize `text` with Cartesia's HTTP `tts/bytes` endpoint and stream the
    audio directly to the speakers as bytes arrive.

    Cartesia's TTS pipeline is natively streaming — bytes start coming back
    while the model is still synthesizing later phonemes — so we bypass the
    Kokoro per-sentence batching pipeline entirely. The full transcript is
    sent in one POST and chunks are piped directly into a sounddevice output
    stream.

    Parameters
    ----------
    text : str
        Full transcript to synthesize. Sent to Cartesia in a single request —
        no sentence chunking on our side.
    speed : float
        Forwarded to Cartesia's `generation_config.speed`.
    play : bool
        When True, play audio through the default output device as bytes
        arrive. When False, the bytes are still consumed (so save can capture
        them) but nothing is sent to the speakers.
    save : bool
        When True, accumulate every byte chunk and write a 16-bit mono WAV
        file at the end of the call. Default False — the request defaults to
        live playback only, never persisting audio to disk.
    save_path : str | None
        Optional override for the output filename. Ignored when save is False.
        When save is True and save_path is None, a timestamped filename is
        used so repeated calls do not clobber each other.
    voice_id : str
        Cartesia voice UUID (passed as `voice.id` in the request body).
    model_id : str
        Cartesia TTS model id (e.g. "sonic-3.5").

    Raises
    ------
    RuntimeError
        If the TTS_KEY environment variable is missing, if the optional
        `requests`/`sounddevice` dependencies cannot be imported, or if
        Cartesia returns a non-2xx response.
    """
    api_key = os.environ.get("TTS_KEY")
    if not api_key:
        raise RuntimeError(
            "Cartesia TTS requested but the TTS_KEY environment variable is "
            "not set. Add TTS_KEY=<your cartesia api key> to your .env file."
        )

    # Lazy imports so the Kokoro-only path of this module never has to load
    # requests/sounddevice. Each is wrapped individually so the error message
    # can name the missing package precisely.
    try:
        import requests  # noqa: PLC0415
    except Exception as exc:
        raise RuntimeError(
            "Cartesia TTS requires the `requests` package, which could not be "
            f"imported. Install it into the project venv. Original error: {exc}"
        ) from exc

    sd = None
    if play:
        try:
            import sounddevice as sd  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError(
                "Cartesia TTS playback requires `sounddevice`, which could "
                f"not be imported. Original error: {exc}"
            ) from exc

    headers = {
        "Cartesia-Version": _CARTESIA_API_VERSION,
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }
    # output_format is fixed to raw / pcm_s16le / 44100 Hz so the bytes coming
    # off the wire are directly playable by sounddevice.RawOutputStream with
    # dtype="int16". WAV container would prepend a 44-byte header that we'd
    # have to skip on the first chunk; raw avoids that complication entirely.
    body = {
        "model_id": model_id,
        "transcript": text,
        "voice": {"mode": "id", "id": voice_id},
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": _CARTESIA_SAMPLE_RATE,
        },
        "language": "en",
        "generation_config": {"speed": speed, "volume": 1},
    }

    # Buffer for the optional WAV save. Even when save=False this stays None
    # so the hot path never allocates per-chunk memory it doesn't need.
    saved_pcm = bytearray() if save else None

    # stream=True tells `requests` to leave the socket open and yield raw byte
    # chunks via iter_content() as they arrive — that's the property that lets
    # the speaker start playing while the rest of the audio is still being
    # synthesized server-side.
    with requests.post(
        _CARTESIA_API_URL,
        headers=headers,
        json=body,
        stream=True,
        timeout=60,
    ) as response:
        if response.status_code >= 400:
            # Pull the full body for the error message — useful for surfacing
            # API-side validation errors (bad voice id, unknown model, etc.).
            try:
                detail = response.text
            except Exception:
                detail = "<no body>"
            raise RuntimeError(
                f"Cartesia TTS request failed: HTTP {response.status_code} — {detail}"
            )

        # Open the playback stream once and feed every chunk into it. We use
        # RawOutputStream because the wire format is raw bytes, not numpy
        # arrays. blocksize=0 lets sounddevice handle internal buffering for
        # arbitrary write sizes; we just need each write to be frame-aligned
        # (mono int16 = 2 bytes per frame).
        playback_stream = None
        if play:
            playback_stream = sd.RawOutputStream(
                samplerate=_CARTESIA_SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=0,
            )
            playback_stream.start()

        # Holds a stray odd byte across iterations on the rare chance that an
        # iter_content chunk does not land on a frame boundary. PCM s16le data
        # is always an even number of bytes overall, so this almost never
        # accumulates content, but the guard keeps RawOutputStream.write()
        # from rejecting an oddly sized buffer.
        leftover = b""

        try:
            for chunk in response.iter_content(chunk_size=_CARTESIA_HTTP_CHUNK):
                if not chunk:
                    continue
                if save:
                    saved_pcm.extend(chunk)
                if playback_stream is not None:
                    data = leftover + chunk
                    even_len = len(data) - (len(data) % 2)
                    if even_len:
                        playback_stream.write(data[:even_len])
                    leftover = data[even_len:]
        finally:
            if playback_stream is not None:
                # stop() blocks until queued audio finishes playing, so the
                # function only returns once the user has actually heard the
                # full response — matching the synchronous contract of the
                # Kokoro path.
                playback_stream.stop()
                playback_stream.close()

    if save and saved_pcm is not None:
        # Wrap the captured raw PCM in a standard WAV container. We do this
        # locally with the stdlib `wave` module instead of asking Cartesia for
        # container=wav so the streaming path stays header-free.
        if save_path is None:
            timestamp = _dt.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
            save_path = f"cartesia_{model_id}_{timestamp}.wav"
        with wave.open(save_path, "wb") as wav_file:
            wav_file.setnchannels(1)        # mono
            wav_file.setsampwidth(2)        # 16-bit (pcm_s16le)
            wav_file.setframerate(_CARTESIA_SAMPLE_RATE)
            wav_file.writeframes(bytes(saved_pcm))


def _generate_deepgram_audio(
    text: str,
    speed: float = 1.0,
    play: bool = True,
    save: bool = False,
    save_path: str | None = None,
    model_id: str = _DEEPGRAM_DEFAULT_MODEL,
) -> None:
    """
    Synthesize `text` with Deepgram's Speak v1 REST API and stream the audio
    directly to the speakers as bytes arrive.

    Like Cartesia, Deepgram begins emitting audio bytes as soon as it has
    something to send — so we send the full transcript in one call, iterate
    `client.speak.v1.audio.generate()` as it yields raw PCM chunks, and feed
    each chunk straight into sounddevice. The Kokoro per-sentence batching
    pipeline is bypassed entirely.

    Parameters
    ----------
    text : str
        Full transcript to synthesize. Sent to Deepgram in a single request.
    play : bool
        When True, play audio through the default output device as bytes
        arrive. When False, the bytes are still consumed (so save can capture
        them) but nothing is sent to the speakers.
    save : bool
        When True, accumulate every byte chunk and write a 16-bit mono WAV
        file at the end of the call. Default False — the request defaults to
        live playback only, never persisting audio to disk.
    save_path : str | None
        Optional override for the output filename. Ignored when save is False.
        When save is True and save_path is None, a timestamped filename is
        used so repeated calls do not clobber each other.
    model_id : str
        Deepgram TTS model id (e.g. "aura-2-hyperion-en").

    Notes
    -----
    `speed` is forwarded to Deepgram's `audio.generate(speed=...)` parameter,
    which is in the same shape as Cartesia's `generation_config.speed`
    (1.0 == real-time). Other Kokoro-shaped arguments on `generate_TTS_audio`
    (`voice`, `lang_code`, `streaming_interval`, `stream`, `model`) are not
    forwarded here because Deepgram exposes voice and language via `model_id`
    (e.g. `aura-2-hyperion-en`) and handles streaming natively.

    Raises
    ------
    RuntimeError
        If DEEPGRAM_TTS_KEY is not set, the deepgram/sounddevice imports
        fail, or the SDK raises during synthesis.
    """
    api_key = os.environ.get("DEEPGRAM_TTS_KEY")
    if not api_key:
        raise RuntimeError(
            "Deepgram TTS requested but the DEEPGRAM_TTS_KEY environment "
            "variable is not set. Add DEEPGRAM_TTS_KEY=<your deepgram api "
            "key> to your .env file."
        )

    # Lazy imports — see the matching pattern in _generate_cartesia_audio for
    # the rationale (keep optional deps off the Kokoro-only import path).
    try:
        from deepgram import DeepgramClient  # noqa: PLC0415
    except Exception as exc:
        raise RuntimeError(
            "Deepgram TTS requires the `deepgram-sdk` package, which could "
            f"not be imported. Install it into the project venv. Original "
            f"error: {exc}"
        ) from exc

    sd = None
    if play:
        try:
            import sounddevice as sd  # noqa: PLC0415
        except Exception as exc:
            raise RuntimeError(
                "Deepgram TTS playback requires `sounddevice`, which could "
                f"not be imported. Original error: {exc}"
            ) from exc

    # Buffer for the optional WAV save. Stays None when save=False so the hot
    # path never allocates per-chunk memory it doesn't need.
    saved_pcm = bytearray() if save else None

    # DeepgramClient v5+ requires `api_key` as a keyword argument — passing
    # it positionally raises TypeError because the underlying BaseClient
    # signature does not accept positional auth.
    client = DeepgramClient(api_key=api_key)
    # encoding="linear16" + sample_rate=24000 means the iterator yields raw
    # mono int16 little-endian PCM at 24 kHz — directly playable by
    # sounddevice.RawOutputStream(samplerate=24000, channels=1, dtype="int16").
    audio_chunks = client.speak.v1.audio.generate(
        text=text,
        model=model_id,
        encoding="linear16",
        sample_rate=_DEEPGRAM_SAMPLE_RATE,
        speed=speed,
    )

    playback_stream = None
    if play:
        playback_stream = sd.RawOutputStream(
            samplerate=_DEEPGRAM_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=0,
        )
        playback_stream.start()

    # Holds a stray odd byte across iterations on the rare chance an SDK
    # chunk does not land on a 2-byte frame boundary. PCM s16le data is always
    # an even number of bytes overall, so this almost never accumulates, but
    # the guard keeps RawOutputStream.write() from rejecting odd buffers.
    leftover = b""

    try:
        for chunk in audio_chunks:
            if not chunk:
                continue
            if save:
                saved_pcm.extend(chunk)
            if playback_stream is not None:
                data = leftover + chunk
                even_len = len(data) - (len(data) % 2)
                if even_len:
                    playback_stream.write(data[:even_len])
                leftover = data[even_len:]
    finally:
        if playback_stream is not None:
            # stop() blocks until queued audio finishes playing, matching the
            # synchronous contract of the Kokoro and Cartesia paths.
            playback_stream.stop()
            playback_stream.close()

    if save and saved_pcm is not None:
        if save_path is None:
            timestamp = _dt.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
            save_path = f"deepgram_{model_id}_{timestamp}.wav"
        with wave.open(save_path, "wb") as wav_file:
            wav_file.setnchannels(1)        # mono
            wav_file.setsampwidth(2)        # 16-bit (linear16 == pcm_s16le)
            wav_file.setframerate(_DEEPGRAM_SAMPLE_RATE)
            wav_file.writeframes(bytes(saved_pcm))


def generate_TTS_audio(
    text="A VLM is an LLM whose input sequence has been extended to include image \
        patches that have been projected into the LLM's embedding space, \
        so attention treats vision and language as one unified token stream.",
    model=_DEFAULT_MODEL_PATH,
    voice="af_sky",
    speed=0.95,
    lang_code="a",
    streaming_interval = 0.5,
    play=True,
    # stream=True tells generate_audio to yield audio chunks incrementally
    # (sentence by sentence) and queue each to AudioPlayer immediately,
    # so playback begins on the first chunk rather than after full synthesis.
    # Without stream=True, generate_audio calls audio_write() on every result
    # regardless of play=True.
    stream=True,
    save=False,
    save_path=None,
):
    if not text or not text.strip():
        raise ValueError("generate_TTS_audio received empty text — nothing to synthesize.")

    # Dispatch on the process-wide voice-model selector set by the LangGraph
    # CLI. Cartesia is intentionally treated as a separate engine — it does
    # NOT consume the Kokoro-shaped `model`/`voice`/`lang_code`/
    # `streaming_interval`/`stream` parameters because Cartesia's HTTP
    # streaming endpoint handles all of those concerns natively. Any caller
    # passing those parameters will simply have them ignored when Cartesia is
    # selected; that is intentional so existing callers (narrate_progress,
    # voice_ask_user, Graph.py greeting/response, NHTSA_Query_Tools.Ask_User)
    # do not need to be edited when the user flips the CLI flag.
    selected = get_voice_model()
    if selected == "cartesia":
        # Cartesia accepts speed as a float multiplier; the Kokoro path uses
        # values like 0.85/0.95/1.1. Pass the same speed straight through —
        # Cartesia's `generation_config.speed` is in the same shape (1.0 == real-time).
        # Guard against zero/negative values that would be invalid in either system.
        cartesia_speed = speed if isinstance(speed, (int, float)) and speed > 0 else 1.0
        _generate_cartesia_audio(
            text=text,
            speed=float(cartesia_speed),
            play=bool(play),
            save=bool(save),
            save_path=save_path,
        )
        return
    if selected == "deepgram":
        # Deepgram's `audio.generate` accepts `speed` (same shape as
        # Cartesia's), so we forward it. The Kokoro-shaped `voice`/
        # `lang_code`/`streaming_interval`/`stream`/`model` arguments are
        # intentionally ignored — Deepgram exposes voice and language via
        # the model id (e.g. `aura-2-hyperion-en`) and handles streaming
        # natively. The dispatch keeps the Kokoro signature stable so
        # existing callers (narrate_progress, voice_ask_user, Graph.py
        # greeting/response, NHTSA_Query_Tools.Ask_User) need no edits.
        deepgram_speed = speed if isinstance(speed, (int, float)) and speed > 0 else 1.0
        _generate_deepgram_audio(
            text=text,
            speed=float(deepgram_speed),
            play=bool(play),
            save=bool(save),
            save_path=save_path,
        )
        return

    # Default Kokoro path — unchanged behavior from the original implementation.
    # If the caller requests the default model, pass the pre-loaded instance to
    # skip load_model() inside generate_audio. If a different model path is given,
    # fall back to the string so generate_audio loads it fresh.
    model_arg = _get_tts_model() if model == _DEFAULT_MODEL_PATH else model

    generate_audio(
        text=text,
        model=model_arg,
        streaming_interval = streaming_interval,
        voice=voice,
        speed=speed,
        lang_code=lang_code,
        play=play,
        stream=stream,
        save=save,
    )


def narrate_progress(text: str) -> None:
    # Faster-paced TTS for progress updates — speed 1.1 vs the default 0.95, and a
    # slightly larger streaming buffer (0.7s) for smoother pacing on short status phrases.
    # Same voice and model as the main TTS for session-wide consistency.
    # Used exclusively by narrate_node_result to announce node completions and loop progress.
    if not text or not text.strip():
        return
    if not is_audio_enabled():
        print(f"[progress] {text}")
        return
    generate_TTS_audio(
        text=text,
        model=_DEFAULT_MODEL_PATH,
        voice="af_sky",
        speed=1.1,
        streaming_interval=0.7,
        play=True,
        stream=True,
        save=False,
    )


def narrate_node_result(node_name: str, context: dict) -> None:
    # Generates a spoken progress update for the user after a node completes.
    # Mercury formats the node output into 2-4 sentences of TTS-safe prose;
    # narrate_progress delivers it at a faster pace than the final-answer voice.
    # The entire call is wrapped in try/except — narration is a UX layer and must
    # never crash the pipeline.
    try:
        # Lazy imports to avoid circular dependencies at module load time.
        # Audio_Playback is imported early in the process; LangGraph.config and Prompts
        # import other project modules, so top-level imports here could cause cycles.
        from LangGraph.config import mercury_llm  # noqa: PLC0415
        from langchain_core.messages import HumanMessage  # noqa: PLC0415
        from Prompts import Node_Progress_Summary_Prompt  # noqa: PLC0415

        prompt_text = Node_Progress_Summary_Prompt(node_name, context)
        response = mercury_llm.invoke([HumanMessage(content=prompt_text)])
        narrate_progress(response.content)
    except Exception as e:
        print(f"[narrate_node_result] narration failed for {node_name}: {e}", file=sys.stderr)