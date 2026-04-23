import sounddevice as sd
import soundfile as sf

import threading
import queue
import numpy as np
import torch
import whisper

# --- Config ---
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512
VAD_THRESHOLD = 0.5
SILENCE_CHUNKS_TO_STOP = 30
WHISPER_BATCH_SECONDS = 3
OUTPUT_WAV = "recording.wav"
OUTPUT_TXT = "transcript.txt"


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


VAD_DEVICE = get_best_device()
WHISPER_DEVICE = get_best_device()
WHISPER_FP16 = WHISPER_DEVICE == "cuda"

# ---------------------------------------------------------------------------
# Whisper model size selector
#
# Available models (larger = more accurate but slower to load/run):
#   "tiny"    — fastest, least accurate (~39 M params)
#   "base"    — good balance for short phrases (~74 M params)       ← default
#   "small"   — noticeably better accuracy (~244 M params)
#   "medium"  — high accuracy, heavier RAM/compute (~769 M params)
#   "large"   — best accuracy, slowest (~1550 M params)
#   "large-v2"— updated large with improved accuracy
#   "large-v3"— latest large, best overall quality
#
# Multi-lingual vs English-only (English-only models end in ".en"):
#   e.g. "tiny.en", "base.en", "small.en", "medium.en"
#   English-only variants are faster and more accurate for English speech.
# ---------------------------------------------------------------------------
WHISPER_MODEL = "small.en"


def validate_whisper_package() -> None:
    whisper_path = getattr(whisper, "__file__", "") or ""
    has_load_model = hasattr(whisper, "load_model")

    if has_load_model:
        return

    raise ImportError(
        "Imported the wrong 'whisper' package from "
        f"{whisper_path or 'an unknown location'}. "
        "This script requires OpenAI Whisper. "
        "Run: pip uninstall whisper && pip install openai-whisper"
    )


validate_whisper_package()

# --- Load models ---
print(f"Loading Silero VAD on {VAD_DEVICE}...")
vad_model, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
vad_model = vad_model.to(VAD_DEVICE)

print(f"Loading Whisper '{WHISPER_MODEL}' on {WHISPER_DEVICE}...")
whisper_model = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)

# --- Shared state ---
# Two separate queues so VAD and Whisper each get every chunk independently.
vad_queue = queue.Queue()     # audio_callback -> vad_worker (stop logic only)
whisper_queue = queue.Queue() # audio_callback -> whisper_worker (all audio)
stop_recording = threading.Event()

all_audio = []  # accumulates everything for the .wav


def vad_check(chunk: np.ndarray) -> float:
    tensor = torch.tensor(chunk, dtype=torch.float32).to(VAD_DEVICE)
    with torch.no_grad():
        return vad_model(tensor, SAMPLE_RATE).item()


def audio_callback(indata, frames, time, status):
    # Keep this callback as lightweight as possible — heavy work (VAD, Whisper)
    # runs in separate threads so we never exceed the ~32ms chunk deadline.
    chunk = indata[:, 0].copy()
    all_audio.append(chunk)   # always accumulate for .wav
    vad_queue.put(chunk)      # VAD worker decides when to stop
    whisper_queue.put(chunk)  # Whisper worker gets everything


def vad_worker():
    # Only responsible for running VAD and triggering stop_recording when
    # sustained silence is detected. Does not gate what Whisper receives.
    silence_count = 0
    speech_detected = False

    while not stop_recording.is_set():
        try:
            chunk = vad_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        is_speech = vad_check(chunk) >= VAD_THRESHOLD

        if is_speech:
            silence_count = 0
            speech_detected = True
        elif speech_detected:
            silence_count += 1
            if silence_count >= SILENCE_CHUNKS_TO_STOP:
                stop_recording.set()


def whisper_worker():
    # Receives all raw audio (not VAD-filtered) and batches it into segments
    # of WHISPER_BATCH_SECONDS for transcription.
    transcript = []
    batch_buffer = []

    while not (stop_recording.is_set() and whisper_queue.empty()):
        try:
            chunk = whisper_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        batch_buffer.extend(chunk)

        # Send a batch to Whisper once enough audio has accumulated, or flush
        # whatever remains when recording stops and the queue is drained.
        batch_ready = len(batch_buffer) >= SAMPLE_RATE * WHISPER_BATCH_SECONDS
        recording_done = stop_recording.is_set() and whisper_queue.empty()

        if batch_buffer and (batch_ready or recording_done):
            segment = np.array(batch_buffer)
            batch_buffer.clear()

            result = whisper_model.transcribe(
                segment,
                fp16=WHISPER_FP16,
                language="en"
            )
            text = result["text"].strip()
            if text:
                transcript.append(text)

    with open(OUTPUT_TXT, "w") as f:
        f.write(" ".join(transcript))
    print(f"Transcript saved to {OUTPUT_TXT}")


# --- Run ---
print("Speak now. Will stop after ~1 second of silence.\n")

vad_thread = threading.Thread(target=vad_worker, daemon=True)
vad_thread.start()

whisper_thread = threading.Thread(target=whisper_worker, daemon=True)
whisper_thread.start()

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    blocksize=CHUNK_SAMPLES,
    callback=audio_callback
):
    stop_recording.wait()

print("Recording stopped, finishing transcription...")
whisper_thread.join()

sf.write(OUTPUT_WAV, np.concatenate(all_audio), SAMPLE_RATE)
print(f"Audio saved to {OUTPUT_WAV}")

print("Playing back...")
data, sr = sf.read(OUTPUT_WAV)
sd.play(data, sr)
sd.wait()
print("Playback complete.")
