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
