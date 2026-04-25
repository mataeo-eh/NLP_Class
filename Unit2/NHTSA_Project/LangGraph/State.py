from typing_extensions import TypedDict


class State(TypedDict):
    user_request: str          # transcribed STT input
    task_type: str             # "analyze", "retrieve", "agentic_retrieve_and_analyze"
    query_result: str          # raw data retrieved from the parquet DB
    reasoning_output: str      # intermediate thinking, exposed to user via TTS
    analysis: str              # labeling / reasoning output from the LLM
    response: str              # final text, ready to be sent to TTS