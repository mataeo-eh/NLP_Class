import json
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from Nodes import classify_task, retrieve_data, analyze, csv_append, agentic_loop
from State import State
from Edges import route_by_task_type, route_after_retrieve
from config import mercury_llm
from Project_Tools.Audio_Playback import generate_TTS_audio
from Project_Tools.Audio_Capture import capture_and_transcribe


def json_to_spoken_text(data: dict | str) -> str:
    """
    Converts the pipeline's output (a State dict or JSON string) into natural
    spoken prose suitable for TTS. Mercury handles this well since it's a fast
    diffusion LLM — good fit for a lightweight formatting step at the end of
    the pipeline.
    """
    # If the data is already a plain string (e.g. response field was already prose),
    # try to parse it as JSON first; if that fails, return it as-is.
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return data

    prompt = (
        "Convert the following data into clear, natural spoken language "
        "suitable for a text-to-speech system. No bullet points, no markdown, "
        "no JSON syntax — just flowing prose a listener can follow.\n\n"
        f"{json.dumps(data, indent=2)}"
    )

    response = mercury_llm.invoke([HumanMessage(content=prompt)])
    return response.content


graph = StateGraph(State)

graph.add_node("classify_task", classify_task)
graph.add_node("retrieve_data", retrieve_data)
graph.add_node("analyze", analyze)
graph.add_node("csv_append", csv_append)
graph.add_node("agentic_loop", agentic_loop)

graph.add_edge(START, "classify_task")   # entry point
# After analysis, persist each per-row result to the prompt's CSV before exit.
# csv_append handles dedup against existing df_index values, so re-running
# analysis on a previously analyzed complaint never produces a duplicate row.
graph.add_edge("analyze", "csv_append")
graph.add_edge("csv_append", END)
graph.add_edge("agentic_loop", END)

# classify_task routes "retrieve" and "analyze" both into retrieve_data,
# since data must be fetched before analysis can run.
# "agentic_retrieve_and_analyze" bypasses retrieve_data and manages its own loop.
graph.add_conditional_edges(
    "classify_task",
    route_by_task_type,
    {
        "retrieve_data": "retrieve_data",
        "agentic_loop":  "agentic_loop",
    }
)

# After retrieve_data, pure retrieval tasks exit; analyze tasks continue onward.
graph.add_conditional_edges(
    "retrieve_data",
    route_after_retrieve,
    {
        "analyze": "analyze",
        "END":     END,
    }
)
app = graph.compile()






if __name__ == "__main__":
    # Greet the user via TTS so they know the system is ready, then capture their
    # spoken request and transcribe it before handing off to the graph.
    generate_TTS_audio(
        text="Hey, welcome back. What are you looking for today?",
        model="mlx-community/Kokoro-82M-bf16",
        voice="af_sky",
        speed=0.85,
        lang_code="a",
        play=True,
        streaming_interval=0.2,
        stream=True,
        save=False,
    )
    user_request = capture_and_transcribe()
    
    result = app.invoke({
    "user_request": user_request,
    "task_type": "",
    "query_result": [],          # list[dict] — populated by retrieve_data
    "reasoning_output": "",
    "analysis": [],              # list[dict] — populated by analyze (per-row results)
    "analysis_prompt_name": "",  # str — populated by analyze, consumed by csv_append
    "response": ""
})

    generate_TTS_audio(
        text=json_to_spoken_text(result["response"]),
        model="mlx-community/Kokoro-82M-bf16",
        voice="af_sky",
        speed=0.85,
        lang_code="a",
        play=True,
        streaming_interval = 0.2,
        # stream=True prevents the library from writing audio to disk.
        # Without it, generate_audio always calls audio_write() regardless of play=True.
        # save defaults to False, so audio is only queued to AudioPlayer, never persisted.
        stream=True,
        save=False,
    )
