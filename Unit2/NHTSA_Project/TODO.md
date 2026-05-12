Test the created demo prompts and ensure they route the system down the correct pathway.
    - Known bugs:
        - the "Analyze" prompt got routed to the agentic retrieve and analyze node.

No longer TTS' the response out of the agent after doing agentic retrieve and analyze. 

Had an error: POST https://nlpclass-production.up.railway.app/run (new conversation 9df81919-41a3-4b32-bb0c-aeaeffeffa66)

[started]
  user_request: Tell me about some of the already identified safety complaints
  continued: false
  session_id: 9df81919-41a3-4b32-bb0c-aeaeffeffa66

[node_update]
  node: classify_task
  delta: task_type: retrieve

[narration]
  channel: progress
  text: The classify task node has just finished running. It received a context that indicates the task type is retrieve.

POST https://nlpclass-production.up.railway.app/audio/speech/stream (deepgram:aura-2-thalia-en)

[error]
  message: EOF when reading a line
  type: EOFError

This happened upon asking the system to retrieve something from one of the CSV files. 