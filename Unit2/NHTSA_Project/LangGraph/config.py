import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def _api_key(name: str) -> str:
    """
    Keep imports and graph compilation working when .env keys are absent.

    A real LLM call still needs the corresponding key; without it, the provider
    will return an authentication error at call time instead of crashing at
    module import time.
    """
    return os.environ.get(name, f"missing-{name.lower()}")

# Shared LLM client for all nodes in this graph.
# Uses Inception Labs' Mercury model, which exposes an OpenAI-compatible API,
# so ChatOpenAI works here with a custom base_url and api_key.
mercury_llm = ChatOpenAI(
    model="mercury-2",
    temperature=0.75,
    api_key=_api_key("INCEPTION_API_KEY"),
    base_url="https://api.inceptionlabs.ai/v1",
)

gpt5_1_llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0.75,
    api_key=_api_key("OPENAI_API_KEY"),
    # no base_url — defaults to https://api.openai.com/v1
)

gpt5_4_mini_llm = ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0.2,
    api_key=_api_key("OPENAI_API_KEY"),
    # no base_url — defaults to https://api.openai.com/v1
)

# OpenRouter exposes an OpenAI-compatible Chat Completions endpoint, so
# ChatOpenAI works with a custom base_url + api_key.
#
# The `extra_body` payload is OpenRouter-specific: it asks the underlying
# model (Tencent Hunyuan 3 preview, free tier) to emit its reasoning trace.
# We pass it via `model_kwargs` because LangChain forwards `model_kwargs`
# into the OpenAI SDK's `chat.completions.create(...)` call, and the SDK
# in turn forwards `extra_body` verbatim into the request JSON — which is
# exactly where OpenRouter looks for the `reasoning` toggle.
#
# IMPORTANT: do NOT use ChatOpenAI's top-level `reasoning=` kwarg here.
# That kwarg routes the request to OpenAI's Responses API, which
# OpenRouter does not implement.
#
# Returned shape (per OpenRouter docs):
#   response.choices[0].message.content           — the assistant text
#   response.choices[0].message.reasoning_details — opaque reasoning blob
# In LangChain, the AIMessage will carry `reasoning_details` inside
# `additional_kwargs` / `response_metadata`. For multi-turn continuity
# OpenRouter requires that blob to be passed back unmodified on the next
# turn's assistant message; if a node in this graph reuses this LLM across
# turns, it must thread `reasoning_details` through the message history.
openrouter_llm = ChatOpenAI(
    model="tencent/hy3-preview:free",
    temperature=0.75,
    api_key=_api_key("OPEN_ROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model_kwargs={"extra_body": {"reasoning": {"enabled": True}}},
) 