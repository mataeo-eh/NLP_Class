from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

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

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    )


vec1_word = "broken" 
vec2_word = "fixed"
# Single word
vec1 = embeddings.embed_query(vec1_word)
# Full sentence
vec2 = embeddings.embed_query(vec2_word)
print("vec1 len",len(vec1))
print("vec2 len", len(vec2))


#print(f"Vec1: {vec1}")
#print(f"Vec2: {vec2}")


import math

def cosine_similarity(vec1, vec2):
    if len(vec1) != len(vec2):
        return "Both vectors much be of the same length"
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0  # avoid division by zero
    
    return dot_product / (norm1 * norm2)




import matplotlib.pyplot as plt

# X-axis = index of embedding dimension
x1 = list(range(len(vec1)))
x2 = list(range(len(vec2)))

plt.figure()

plt.plot(x1, vec1, label=f"{vec1_word} embedding")
plt.plot(x2, vec2, label=f"{vec2_word} embedding")

plt.title("Embedding Values by Dimension Index")
plt.xlabel("Dimension index")
plt.ylabel("Embedding value")
plt.legend()
plt.show()

print("\n")
print("="*40)
print("Cosine Similarity score:")
print(cosine_similarity(vec1, vec2))
print("="*40)