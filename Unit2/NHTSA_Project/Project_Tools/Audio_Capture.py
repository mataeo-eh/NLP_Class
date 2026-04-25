"""
Audio_Capture.py
----------------
Importable module that captures microphone input, uses Silero VAD to detect
end-of-speech, and transcribes the recorded audio with OpenAI Whisper.

Public API
----------
capture_and_transcribe() -> str
    Records the user's spoken input, transcribes it, and returns the plain-text
    transcript. Nothing is written to disk. Safe to call multiple times — each
    call creates its own fresh state (queues, events, etc.) while reusing the
    module-level VAD and Whisper models that were loaded once at import time.

Design notes
------------
- VAD (Silero) only decides *when* to stop recording. It does not gate what
  Whisper receives — all raw audio goes to Whisper so no speech is dropped.
- Whisper processes audio in rolling batches (WHISPER_BATCH_SECONDS) rather
  than one giant segment, so transcription latency is lower for long inputs.
- Models are loaded once at module import time and kept in memory. Cold-start
  cost is paid the first time this module is imported, not on every call.
"""

import sounddevice as sd
import threading
import queue
import numpy as np
import torch
import whisper


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000          # Hz — required by Silero VAD
CHUNK_SAMPLES = 512          # ~32 ms per chunk at 16 kHz
VAD_THRESHOLD = 0.5          # Silero speech-probability cutoff
SILENCE_CHUNKS_TO_STOP = 50  # ~1.6 s of silence triggers stop
WHISPER_BATCH_SECONDS = 3    # audio batched to Whisper in 3-second segments


def _get_best_device() -> str:
    """Return the best available torch device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_VAD_DEVICE = _get_best_device()
_WHISPER_DEVICE = _get_best_device()

# fp16 is only supported on CUDA; MPS and CPU must use fp32
_WHISPER_FP16 = _WHISPER_DEVICE == "cuda"

# ---------------------------------------------------------------------------
# Whisper model size selector
#
# Available models (larger = more accurate but slower to load/run):
#   "tiny"     — fastest, least accurate (~39 M params)
#   "base"     — good balance for short phrases (~74 M params)
#   "small"    — noticeably better accuracy (~244 M params)      ← default
#   "medium"   — high accuracy, heavier RAM/compute (~769 M params)
#   "large-v3" — best overall quality, slowest
#
# Append ".en" for English-only variants (faster and more accurate for English):
#   e.g. "tiny.en", "base.en", "small.en", "medium.en"
# ---------------------------------------------------------------------------
_WHISPER_MODEL_NAME = "small.en"


def _validate_whisper_package() -> None:
    """Raise ImportError if the 'whisper' import resolved to the wrong package."""
    if hasattr(whisper, "load_model"):
        return
    raise ImportError(
        f"Imported the wrong 'whisper' package from "
        f"{getattr(whisper, '__file__', 'unknown location')}. "
        "This module requires OpenAI Whisper. "
        "Run: pip uninstall whisper && pip install openai-whisper"
    )


_validate_whisper_package()

# ---------------------------------------------------------------------------
# Model loading — happens once at import time, reused across all calls
# ---------------------------------------------------------------------------

print(f"[Audio_Capture] Loading Silero VAD on {_VAD_DEVICE}...")
_vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
)
_vad_model = _vad_model.to(_VAD_DEVICE)

print(f"[Audio_Capture] Loading Whisper '{_WHISPER_MODEL_NAME}' on {_WHISPER_DEVICE}...")
# whisper.load_model(..., device="mps") fails because Whisper's model contains
# a sparse COO buffer (alignment_heads) used only for word-level timestamps.
# MPS doesn't support sparse tensor ops, so .to("mps") raises NotImplementedError.
#
# Fix: load on CPU, then use _apply to move every tensor to MPS individually,
# skipping any sparse buffers (they stay on CPU where sparse ops are supported).
# alignment_heads is only accessed when word_timestamps=True — we don't use that,
# so leaving it on CPU is safe and has no effect on normal transcription.
_whisper_model = whisper.load_model(_WHISPER_MODEL_NAME, device="cpu")
if _WHISPER_DEVICE != "cpu":
    _whisper_model._apply(
        lambda t: t.to(_WHISPER_DEVICE) if not t.is_sparse else t
    )


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _vad_check(chunk: np.ndarray) -> float:
    """Return Silero's speech probability (0–1) for a single audio chunk."""
    tensor = torch.tensor(chunk, dtype=torch.float32).to(_VAD_DEVICE)
    with torch.no_grad():
        return _vad_model(tensor, SAMPLE_RATE).item()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def capture_and_transcribe() -> str:
    """
    Open the microphone, record until VAD detects end-of-speech, transcribe
    via Whisper, and return the transcript as a plain string.

    No audio or transcript is written to disk. The function blocks until
    speech has been detected and a period of silence follows.

    Returns
    -------
    str
        The Whisper transcript of the user's spoken input. Returns an empty
        string if no speech was detected.
    """
    # Fresh per-call state — safe to call this function multiple times
    vad_q: queue.Queue = queue.Queue()
    whisper_q: queue.Queue = queue.Queue()
    stop_evt = threading.Event()
    transcript_result: list[str] = []  # whisper_worker writes the final string here

    # ------------------------------------------------------------------
    # Audio callback — called on the sounddevice I/O thread every chunk.
    # Must be as lightweight as possible to avoid missing the ~32ms deadline.
    # ------------------------------------------------------------------
    def _audio_callback(indata, _frames, _time, _status):
        chunk = indata[:, 0].copy()
        vad_q.put(chunk)     # VAD worker decides when to stop
        whisper_q.put(chunk) # Whisper worker transcribes everything

    # ------------------------------------------------------------------
    # VAD worker — only responsible for detecting end-of-speech and
    # setting stop_evt. Does not filter what Whisper receives.
    # ------------------------------------------------------------------
    def _vad_worker():
        silence_count = 0
        speech_detected = False

        while not stop_evt.is_set():
            try:
                chunk = vad_q.get(timeout=0.5)
            except queue.Empty:
                continue

            is_speech = _vad_check(chunk) >= VAD_THRESHOLD

            if is_speech:
                silence_count = 0
                speech_detected = True
            elif speech_detected:
                silence_count += 1
                if silence_count >= SILENCE_CHUNKS_TO_STOP:
                    stop_evt.set()

    # ------------------------------------------------------------------
    # Whisper worker — receives all raw audio, batches into segments of
    # WHISPER_BATCH_SECONDS, transcribes each, then writes the joined
    # transcript into transcript_result once the queue is fully drained.
    # ------------------------------------------------------------------
    def _whisper_worker():
        transcript: list[str] = []
        batch_buffer: list[float] = []

        while not (stop_evt.is_set() and whisper_q.empty()):
            try:
                chunk = whisper_q.get(timeout=0.5)
            except queue.Empty:
                continue

            batch_buffer.extend(chunk)

            batch_ready = len(batch_buffer) >= SAMPLE_RATE * WHISPER_BATCH_SECONDS
            recording_done = stop_evt.is_set() and whisper_q.empty()

            if batch_buffer and (batch_ready or recording_done):
                segment = np.array(batch_buffer)
                batch_buffer.clear()
                result = _whisper_model.transcribe(
                    segment, fp16=_WHISPER_FP16, language="en"
                )
                text = result["text"].strip()
                if text:
                    transcript.append(text)

        # Flush any audio that accumulated after the last full batch was sent
        # but before stop_evt was set (stop_evt can fire mid-timeout window).
        if batch_buffer:
            segment = np.array(batch_buffer)
            result = _whisper_model.transcribe(
                segment, fp16=_WHISPER_FP16, language="en"
            )
            text = result["text"].strip()
            if text:
                transcript.append(text)

        transcript_result.append(" ".join(transcript))

    # ------------------------------------------------------------------
    # Start workers and open the mic stream
    # ------------------------------------------------------------------
    vad_thread = threading.Thread(target=_vad_worker, daemon=True)
    whisper_thread = threading.Thread(target=_whisper_worker, daemon=True)
    vad_thread.start()
    whisper_thread.start()

    print("[Audio_Capture] Listening... (will stop after ~1 s of silence)")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        callback=_audio_callback,
    ):
        stop_evt.wait()  # blocks until VAD signals end-of-speech

    print("[Audio_Capture] Recording stopped, finishing transcription...")
    whisper_thread.join()

    return transcript_result[0] if transcript_result else ""
