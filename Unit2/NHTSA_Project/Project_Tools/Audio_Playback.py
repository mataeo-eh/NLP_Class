import sys
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


def narrate_progress(text: str) -> None:
    # Faster-paced TTS for progress updates — speed 1.1 vs the default 0.95, and a
    # slightly larger streaming buffer (0.7s) for smoother pacing on short status phrases.
    # Same voice and model as the main TTS for session-wide consistency.
    # Used exclusively by narrate_node_result to announce node completions and loop progress.
    if not text or not text.strip():
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
