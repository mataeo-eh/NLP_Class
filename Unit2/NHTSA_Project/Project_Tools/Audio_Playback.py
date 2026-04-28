from dotenv import load_dotenv
from mlx_audio.tts.generate import generate_audio
from mlx_audio.tts.utils import load_model

load_dotenv()

# Pre-load the Kokoro model once at module import time so it sits in memory
# and is reused across all generate_TTS_audio calls. Loading takes several
# seconds; doing it here means the first TTS call has no cold-start penalty.
# lazy=False forces all weights into memory immediately rather than on first use.
_DEFAULT_MODEL_PATH = "mlx-community/Kokoro-82M-bf16"
_tts_model = load_model(model_path=_DEFAULT_MODEL_PATH, lazy=False)


def generate_TTS_audio(
    text="Goodbye... cruel... world... I, I move towards the light.",
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
):
    if not text or not text.strip():
        raise ValueError("generate_TTS_audio received empty text — nothing to synthesize.")

    # If the caller requests the default model, pass the pre-loaded instance to
    # skip load_model() inside generate_audio. If a different model path is given,
    # fall back to the string so generate_audio loads it fresh.
    model_arg = _tts_model if model == _DEFAULT_MODEL_PATH else model

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
