<objective>
Add an audio progress narration system to the NHTSA LangGraph pipeline. The system gives the user real-time spoken feedback after each node completes and during agentic tool-calling loops, so they know the pipeline is active rather than stalled. Mercury (the diffusion LLM already wired into the pipeline) generates concise spoken summaries; Kokoro TTS delivers them at a faster pace than the final-result voice.
</objective>

<context>
Project root: Unit2/NHTSA_Project/
Pipeline flow: classify_task → (classify_agentic_subtype →) retrieve_data / analyze / agentic_analyze / agentic_explore → csv_append

Key files and what they contain:

Unit2/NHTSA_Project/Project_Tools/Audio_Playback.py
  - generate_TTS_audio(text, model, voice="af_sky", speed=0.95, lang_code="a",
                       streaming_interval=0.5, play=True, stream=True, save=False)
  - _DEFAULT_MODEL_PATH = "mlx-community/Kokoro-82M-bf16"
  - _tts_model: pre-loaded Kokoro instance (reused when default path is passed)
  - streaming_interval: seconds of audio buffered before playback begins

Unit2/NHTSA_Project/Prompts.py
  - Contains all LLM system/user prompt builders for the pipeline
  - Existing analysis prompts: Safety_Prompt, Specific_Subsystem_Prompt
  - Existing task prompts: Specify_Task_Type, Specify_Agentic_Subtype, Retrieve_Data_Prompt, Agentic_Analyze_Prompt, Agentic_Explore_Prompt
  - get_available_prompts() returns the list of analysis prompts for user selection
  - ANALYSIS_PROMPTS dict maps prompt names to (function, description) tuples

Unit2/NHTSA_Project/LangGraph/config.py — defines the three shared LLM clients:
  - mercury_llm = ChatOpenAI(model="mercury-2", base_url="https://api.inceptionlabs.ai/v1")
    Inception Labs' Mercury diffusion LLM, exposed via OpenAI-compatible API.
    Called via: mercury_llm.invoke([HumanMessage(content=prompt)]) → AIMessage with .content str
  - gpt5_1_llm   = ChatOpenAI(model="gpt-5.1")    — strong reasoning, used by analyze + agentic_analyze
  - gpt5_4_mini_llm = ChatOpenAI(model="gpt-5.4-mini") — fast, used by agentic_explore
  Import mercury_llm as: from LangGraph.config import mercury_llm
  (confirm the import style against existing Nodes.py imports — use whatever relative/absolute
  path pattern the project already uses for LangGraph/ imports)

Unit2/NHTSA_Project/LangGraph/State.py — State TypedDict fields:
  user_request: str
  task_type: str           ("analyze" | "retrieve" | "agentic_retrieve_and_analyze")
  query_result: list[dict]
  reasoning_output: str    — already in State for intermediate thinking exposed via TTS
  analysis: list[dict]     — per-row results from the analyze node
  analysis_prompt_name: str
  response: str
  agentic_subtype: str     ("agentic_analyze" | "agentic_explore" | "")
  iteration_log: list[dict] — one entry per agentic loop iteration:
    {"iter": int, "model": str, "tool_calls": list[str], "summary": str}

Unit2/NHTSA_Project/LangGraph/Nodes.py — graph node functions:
  - classify_task(state) ~line 70
  - classify_agentic_subtype(state) ~line 92
  - retrieve_data(state) ~line 147
  - analyze(state) ~line 253  (two-step: GPT-5.1 picks prompt, GPT-5.4-mini runs it per row)
  - agentic_analyze(state) ~line 472  (bounded loop, MAX_ITERATIONS=12)
  - agentic_explore(state) ~line 631  (bounded loop, MAX_ITERATIONS=12)
</context>

<requirements>
1. NARRATION TTS WRAPPER — add `narrate_progress(text: str) -> None` to Audio_Playback.py

   Parameters versus generate_TTS_audio defaults:
   - speed: 1.1  (faster than the default 0.95 — progress updates should feel snappy)
   - streaming_interval: 0.7  (slightly larger buffer = smoother pacing for short status phrases;
     if the audio is so short that Kokoro only produces 0.5-second chunks, the library handles
     that gracefully — no special fallback logic needed in Python)
   - voice: "af_sky"  (same as default — voice consistency across the session)
   - model: _DEFAULT_MODEL_PATH  (reuse the pre-loaded _tts_model instance)
   - play=True, stream=True, save=False  (same as main TTS)

   Comment explaining the rationale: faster speed + larger streaming buffer = quick, clear
   status narration without the slower, more deliberate pace of the final-answer voice.

2. MERCURY PROGRESS PROMPT — add `Node_Progress_Summary_Prompt(node_name: str, context: dict) -> str`
   to Prompts.py

   Purpose: formats a single HumanMessage prompt string (NOT a system/user dict pair — just plain
   text) that tells Mercury to produce a short spoken progress update.

   The `context` dict may contain any subset of these keys:
     task_type (str)           — e.g. "agentic_retrieve_and_analyze"
     agentic_subtype (str)     — e.g. "agentic_analyze"
     tool_calls (list[str])    — names of tools called in the current iteration
     reasoning (str)           — one-sentence summary of what happened this iteration
     analysis_prompt_name (str)— e.g. "Safety_Prompt"
     row_count (int)           — number of rows retrieved or analyzed
     classification (str)      — classification verdict for the current row/batch
     iter (int)                — current iteration number
     max_iter (int)            — iteration cap for the current loop

   Prompt requirements:
   - Tell Mercury: output 2 to 4 sentences of clear spoken prose
   - No bullet points, no markdown, no JSON, no ellipses (TTS cannot pronounce them naturally),
     no parenthetical asides — flowing natural speech only
   - Include the node_name and all provided context keys as labeled inputs in the prompt so
     Mercury has full visibility into what just happened
   - Example framing to include in the prompt text:
       "Node: classify_task | task_type: agentic_retrieve_and_analyze"
       → Mercury should output something like: "I've classified your request as an agentic
         retrieval and analysis task. I'll retrieve relevant complaints and reason over them
         to answer your question."
   - CAPITALIZATION NORMALIZATION — explicitly instruct Mercury in the prompt:
       "Normalize all capitalization in your output for natural text-to-speech playback.
        Convert any ALL-CAPS words (e.g. TOYOTA, CAMRY, NHTSA, BRAKES) to title case or
        sentence case as appropriate. The TTS tokenizer verbalizes all-caps words
        character-by-character rather than as words, so 'TOYOTA' becomes 'T-O-Y-O-T-A'
        unless normalized. Apply this to every proper noun, vehicle name, acronym, and
        dataset field value that appears in your output."
   - Add a comment explaining: Mercury is used here as a lightweight spoken-language formatter,
     not a reasoning engine. One HumanMessage call, no system prompt, fast turnaround.
     Capitalization normalization is critical because upstream models output high-fidelity
     all-caps strings (e.g. "TOYOTA CAMRY", "ELECTRICAL SYSTEM") that the Kokoro TTS
     tokenizer cannot handle — it spells them out letter by letter instead of speaking them.

   Place this function in Prompts.py after the existing imports/constants block and BEFORE
   get_available_prompts, since it is a utility prompt not an analysis prompt.

3. NARRATION UTILITY — add `narrate_node_result(node_name: str, context: dict) -> None`
   to Audio_Playback.py

   Implementation:
   a. Import mercury_llm from LangGraph.config (confirmed: defined at config.py line 10)
   b. Import HumanMessage from langchain_core.messages
   c. Import Node_Progress_Summary_Prompt from Prompts (be careful of circular imports —
      Audio_Playback does not import from LangGraph/, so importing from Prompts.py is safe)
   d. Call Node_Progress_Summary_Prompt(node_name, context) to get the prompt string
   e. Call mercury_llm.invoke([HumanMessage(content=prompt_text)]) to get spoken prose
   f. Pass response.content to narrate_progress(response.content)
   g. Wrap steps d–f in try/except Exception as e:
        print(f"[narrate_node_result] narration failed for {node_name}: {e}", file=sys.stderr)
        return
      Rationale: narration is a UX layer — it must never crash the pipeline. If Mercury
      is slow, unavailable, or returns garbage, the pipeline continues silently.

4. NODE HOOKS — add narrate_node_result calls in Nodes.py at these exact points
   (add the import at the top of Nodes.py alongside existing imports):

   from Unit2.NHTSA_Project.Project_Tools.Audio_Playback import narrate_node_result
   (use whatever import style Nodes.py already uses for Project_Tools imports)

   Hook locations:

   a. classify_task — after task_type is determined, before returning state dict:
      narrate_node_result("classify_task", {"task_type": task_type})

   b. classify_agentic_subtype — after agentic_subtype is set:
      narrate_node_result("classify_agentic_subtype", {"agentic_subtype": agentic_subtype})

   c. retrieve_data — after the retrieval loop exits and results are collected:
      narrate_node_result("retrieve_data", {
          "row_count": len(query_result),
          "task_type": state.get("task_type", "")
      })

   d. analyze — two hooks:
      i.  After the analysis prompt is selected (step 1, GPT-5.1 call):
          narrate_node_result("analyze_prompt_selected", {
              "analysis_prompt_name": chosen_prompt_name
          })
      ii. After all rows are processed (step 2 loop complete):
          narrate_node_result("analyze_complete", {
              "row_count": len(analysis_results),
              "analysis_prompt_name": chosen_prompt_name
          })

   e. agentic_analyze AND agentic_explore — inside the iteration loop, at the END of each
      iteration, after the iteration_log entry is appended:
      narrate_node_result("agentic_iteration", {
          "iter": current_iter,
          "max_iter": MAX_ITERATIONS,
          "tool_calls": tool_calls_this_iter,   # list of tool name strings called this turn
          "reasoning": summary_this_iter         # the one-sentence summary already computed
                                                 # for the iteration_log entry
      })
      Note: Kokoro's AudioPlayer queues audio in a background thread, so TTS plays
      concurrently with the next LLM call. This hook does not block the loop.
</requirements>

<constraints>
- Do NOT change generate_TTS_audio's existing signature or default parameter values.
  narrate_progress is a separate, additive function.
- Do NOT add narration to csv_append — it is a pure I/O operation.
- narrate_node_result must never raise an exception. The try/except must be broad
  (except Exception) and must print to stderr, not stdout.
- The Node_Progress_Summary_Prompt output that Mercury receives must be TTS-safe:
  no ellipses, no markdown symbols, no JSON brackets, no numbered lists.
- Check for circular imports before writing the import in Audio_Playback.py.
  Audio_Playback.py must not import from LangGraph/ modules.
- mercury_llm is defined in Unit2/NHTSA_Project/LangGraph/config.py — import it as
  `from LangGraph.config import mercury_llm` (or match the existing import style in Nodes.py).
  Do not import from Graph.py.
</constraints>

<implementation_steps>
1. Read the first 40 lines of Unit2/NHTSA_Project/LangGraph/Nodes.py to confirm the
   existing import style for LangGraph.config and Project_Tools modules.
2. Read the first 25 lines of Unit2/NHTSA_Project/Project_Tools/Audio_Playback.py
   to understand existing imports.
3. Add narrate_progress to Audio_Playback.py (after generate_TTS_audio).
4. Add Node_Progress_Summary_Prompt to Prompts.py (after constants, before get_available_prompts).
5. Add narrate_node_result to Audio_Playback.py (after narrate_progress), importing
   mercury_llm from LangGraph.config.
6. Add the narrate_node_result import to Nodes.py.
7. Add the six hook call sites to Nodes.py at the locations specified in requirement 4.
8. Run verification checks.
</implementation_steps>

<verification>
Run these checks after implementing:

1. grep -n "narrate_progress\|narrate_node_result" Unit2/NHTSA_Project/Project_Tools/Audio_Playback.py
   Expected: both functions defined

2. grep -n "narrate_node_result" Unit2/NHTSA_Project/LangGraph/Nodes.py
   Expected: 1 import line + at least 6 call sites

3. grep -n "Node_Progress_Summary_Prompt" Unit2/NHTSA_Project/Prompts.py
   Expected: function definition present

4. grep -n "Node_Progress_Summary_Prompt" Unit2/NHTSA_Project/Project_Tools/Audio_Playback.py
   Expected: import line + 1 call inside narrate_node_result

5. python -m py_compile Unit2/NHTSA_Project/Project_Tools/Audio_Playback.py
   python -m py_compile Unit2/NHTSA_Project/Prompts.py
   python -m py_compile Unit2/NHTSA_Project/LangGraph/Nodes.py
   Expected: all exit 0 with no output
</verification>

<success_criteria>
- narrate_progress in Audio_Playback.py wraps generate_TTS_audio with speed=1.1, streaming_interval=0.7
- Node_Progress_Summary_Prompt in Prompts.py builds a TTS-safe HumanMessage string directing Mercury
  to summarise node output as 2-4 sentences of natural spoken prose
- narrate_node_result in Audio_Playback.py: calls Mercury via the prompt, passes the result to
  narrate_progress, and never raises — any failure is logged to stderr and swallowed
- Six narration hook sites in Nodes.py: classify_task, classify_agentic_subtype, retrieve_data,
  analyze (prompt selection), analyze (completion), and inside the agentic loop per iteration
- All py_compile checks pass with exit code 0
</success_criteria>
