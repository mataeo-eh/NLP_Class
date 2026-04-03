"""
LiteLLM API Call Reference
===========================
Pulls API key, base URL, and model from environment variables.
Every supported completion() parameter is shown and explained below.

Required env vars:
    LITELLM_API_KEY   - Your API key for the provider
    LITELLM_API_BASE  - The base URL of the LiteLLM proxy or provider endpoint
    LITELLM_MODEL     - The model string, e.g. "openai/gpt-4o" or "anthropic/claude-3-5-sonnet"
"""

import os
import litellm
from litellm import completion
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Load environment variables from a .env file
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# Environment variable guards
# ---------------------------------------------------------------------------
try:
    api_key = os.environ["LITELLM_API_KEY"]
except KeyError:
    raise EnvironmentError(
        "LITELLM_API_KEY is not set. "
        "Export your provider API key before running this script, e.g.:\n"
        "  export LITELLM_API_KEY='sk-...'"
    )

try:
    api_base = os.environ["LITELLM_API_BASE"]
except KeyError:
    raise EnvironmentError(
        "LITELLM_API_BASE is not set. "
        "Export the base URL of your LiteLLM proxy or provider endpoint, e.g.:\n"
        "  export LITELLM_API_BASE='http://localhost:4000'"
    )

try:
    model = os.environ["LITELLM_MODEL"]
except KeyError:
    raise EnvironmentError(
        "LITELLM_MODEL is not set. "
        "Export the model identifier in 'provider/model-name' format, e.g.:\n"
        "  export LITELLM_MODEL='openai/gpt-4o'"
    )


# ---------------------------------------------------------------------------
# Conversation messages
# ---------------------------------------------------------------------------
# messages is the only truly required field alongside model.
# It is a list of dicts, each with at minimum "role" and "content".
#
# Roles:
#   "system"    - Sets the assistant's behavior/persona for the whole session.
#   "user"      - Input from the human turn.
#   "assistant" - Prior model responses (for multi-turn conversations).
#   "function"  - Result of a function/tool call returned to the model.
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.",
        # "name": "optional_author_name"   # Max 64 chars. Required when role="function".
    },
    {
        "role": "user",
        "content": "Explain large language models in one paragraph.",
    },
]

# ---------------------------------------------------------------------------
# Tool / function definitions (optional)
# ---------------------------------------------------------------------------
# Lets the model call structured functions and return JSON arguments for them.
# The model does NOT execute functions itself — it signals which one to call
# and with what args; your code handles the actual execution.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            # description: Tells the model what the function does so it can
            # decide when to invoke it.
            "description": "Get the current weather for a given city.",
            # parameters: JSON Schema describing the expected arguments.
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Boston'",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

# ---------------------------------------------------------------------------
# Full completion() call with every supported parameter
# ---------------------------------------------------------------------------
response = completion(
    # ------------------------------------------------------------------
    # REQUIRED
    # ------------------------------------------------------------------

    model=model,
    # Which model to call. LiteLLM uses a "provider/model-name" format,
    # e.g. "openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022",
    # "ollama/llama3". The provider prefix tells LiteLLM which SDK adapter
    # to use under the hood.

    messages=messages,
    # The conversation history. See the `messages` list above.

    # ------------------------------------------------------------------
    # ROUTING / AUTH (LiteLLM-specific)
    # ------------------------------------------------------------------

    api_key=api_key,
    # The API key sent to the upstream provider. Overrides any key already
    # set via environment variable for this single call. Useful when
    # rotating keys or calling multiple providers in the same script.

    api_base=api_base,
    # The base URL of the endpoint to hit. Required when pointing at a
    # self-hosted LiteLLM proxy or a non-default provider URL.
    # Example: "http://localhost:4000"

    # ------------------------------------------------------------------
    # SAMPLING & CREATIVITY
    # ------------------------------------------------------------------

    temperature=0.7,
    # Controls randomness. Range: 0.0 – 2.0.
    # - 0.0 → nearly deterministic; the highest-probability token is
    #         almost always chosen. Good for factual/structured tasks.
    # - 1.0 → the model's default behavior, balanced creativity.
    # - 2.0 → very random/creative; can become incoherent.
    # Do NOT use together with top_p (pick one).

    #top_p=0.9,
    # Nucleus sampling. The model only considers tokens whose cumulative
    # probability reaches top_p. Range: 0.0 – 1.0.
    # - 0.9 → uses the top 90% probability mass; ignores rare tail tokens.
    # - 1.0 → consider all tokens (no filtering).
    # Do NOT use together with temperature (pick one).

    #top_k=40,
    # Hard-limits sampling to the top-k most probable tokens at each step.
    # Not supported by all providers (OpenAI ignores it; local/OSS models
    # like Ollama and Replicate support it).
    # - Lower k → more focused; higher k → more varied.

    # ------------------------------------------------------------------
    # LENGTH CONTROL
    # ------------------------------------------------------------------

    max_tokens=8000,
    # Maximum number of tokens the model may generate in its response.
    # Does NOT count prompt tokens. The call ends when either:
    #   (a) the model emits a stop token, or
    #   (b) this limit is reached (finish_reason will be "length").
    # Set conservatively to control cost and latency.

    # ------------------------------------------------------------------
    # STOPPING CONDITIONS
    # ------------------------------------------------------------------

    #stop=["\n\n", "###"],
    # Up to 4 strings that tell the model to stop generating immediately
    # when it produces any of them. The stop string itself is NOT included
    # in the output. Useful for structured generation (e.g., stopping at
    # a delimiter or section header).

    # ------------------------------------------------------------------
    # REPETITION CONTROL
    # ------------------------------------------------------------------

    #presence_penalty=0.0,
    # Penalizes tokens that have appeared *at all* in the text so far,
    # regardless of frequency. Range: -2.0 – 2.0.
    # - Positive → discourages the model from mentioning any topic twice.
    # - Negative → encourages the model to revisit topics.
    # Good for keeping responses on new ground.

    #frequency_penalty=0.0,
    # Penalizes tokens proportional to *how often* they've appeared.
    # Range: -2.0 – 2.0.
    # - Positive → suppresses verbatim repetition of phrases.
    # - Negative → allows the model to repeat itself more freely.

    #repetition_penalty=1.0,
    # Alternative repetition control used by some OSS/local models
    # (Ollama, Replicate, Bytez). Values > 1.0 penalize repetition;
    # 1.0 = no penalty. Not the same scale as frequency_penalty.

    # ------------------------------------------------------------------
    # MULTI-CHOICE GENERATION
    # ------------------------------------------------------------------

    n=1,
    # Number of independent completion choices to generate for this
    # prompt. Each choice is a separate response from the same model.
    # Useful for best-of-n selection or diversity sampling.
    # Increases cost/latency proportionally.

    # ------------------------------------------------------------------
    # STRUCTURED / CONSTRAINED OUTPUT
    # ------------------------------------------------------------------

    response_format={"type": "text"},
    # Controls the output format.
    # - {"type": "text"}        → plain text (default)
    # - {"type": "json_object"} → forces the model to return valid JSON.
    #                             Your system prompt should instruct the
    #                             model to produce JSON when using this.

    # ------------------------------------------------------------------
    # TOKEN PROBABILITY TWEAKING
    # ------------------------------------------------------------------

    #logit_bias={},
    # A dict mapping token IDs (as strings) to a bias value (-100 – 100).
    # - +100 → virtually guarantees the token is chosen.
    # - -100 → virtually bans the token.
    # Example: {"15043": 10} biases the token for "Hello" upward.
    # Use tokenizer tools to find token IDs. Rarely needed in practice.

    #logprobs=False,
    # If True, the response includes log-probabilities for each generated
    # token. Useful for uncertainty estimation, re-ranking, or analysis.
    # Increases response payload size.

    # ------------------------------------------------------------------
    # TOOL / FUNCTION CALLING
    # ------------------------------------------------------------------

    tools=tools,
    # List of tool/function schemas the model may call. See the `tools`
    # definition above. When the model decides to invoke a tool, the
    # response's finish_reason will be "tool_calls" and the content will
    # contain the function name + JSON arguments.

    tool_choice="auto",
    # Controls whether and which tool the model uses.
    # - "none"      → model will never call a tool (plain text only).
    # - "auto"      → model decides whether to call a tool or respond normally.
    # - {"type": "function", "function": {"name": "get_weather"}}
    #               → forces the model to call exactly that function.

    # ------------------------------------------------------------------
    # REPRODUCIBILITY
    # ------------------------------------------------------------------

    #seed=42,
    # When set, the model attempts deterministic sampling (same seed +
    # same inputs → same output). Not guaranteed by all providers, but
    # best-effort on OpenAI and many others. Useful for testing and
    # reproducibility.

    # ------------------------------------------------------------------
    # STREAMING
    # ------------------------------------------------------------------

    stream=False,
    # If True, the API returns a generator that yields partial message
    # deltas as they are produced (Server-Sent Events). The final chunk
    # carries finish_reason. Set to True for real-time UIs or long
    # responses where you want to display text as it arrives.

    # ------------------------------------------------------------------
    # USER TRACKING
    # ------------------------------------------------------------------

    user="student-demo",
    # An opaque string identifying the end-user making the request.
    # Passed through to the provider for abuse monitoring and per-user
    # analytics. Does not affect model output.

    # ------------------------------------------------------------------
    # TIMEOUT & RELIABILITY (LiteLLM-specific)
    # ------------------------------------------------------------------

    timeout=60,
    # Seconds before LiteLLM raises an APITimeoutError if the provider
    # has not responded. Prevents the script from hanging indefinitely.
    # Catch with: except openai.APITimeoutError.

    # ------------------------------------------------------------------
    # DEBUGGING (LiteLLM-specific)
    # ------------------------------------------------------------------

    drop_params=True,
    # When True, LiteLLM silently drops any parameter that the target
    # provider does not support, instead of raising an error. Useful when
    # writing provider-agnostic code that passes all params regardless of
    # the backend. Commented out here so unsupported params surface as errors during development.
)

# ---------------------------------------------------------------------------
# Parse and print the response
# ---------------------------------------------------------------------------
# response follows the OpenAI ChatCompletion schema regardless of provider.

choice         = response.choices[0]
content        = choice.message.content       # The model's text reply
finish_reason  = choice.finish_reason         # "stop" | "length" | "tool_calls" | "content_filter"

usage          = response.usage
prompt_tokens  = usage.prompt_tokens          # Tokens consumed by the prompt
completion_tok = usage.completion_tokens      # Tokens generated by the model
total_tokens   = usage.total_tokens           # prompt + completion

print("=== Response ===")
print(content)
print()
print(f"Finish reason : {finish_reason}")
print(f"Prompt tokens : {prompt_tokens}")
print(f"Completion tok: {completion_tok}")
print(f"Total tokens  : {total_tokens}")
