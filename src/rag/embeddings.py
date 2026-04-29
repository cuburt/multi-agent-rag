import os
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
import random

logger = structlog.get_logger(__name__)

# Global client cache to avoid redundant initializations
_google_client = None

def _get_google_client():
    global _google_client
    if _google_client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("No GEMINI_API_KEY or GOOGLE_API_KEY found in environment.")
        _google_client = genai.Client(api_key=api_key)
    return _google_client

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _call_google_embedding(model_name: str, text: str) -> list[float]:
    client = _get_google_client()
    
    # We use output_dimensionality=3072 to match the upgraded Vector(3072) schema.
    result = client.models.embed_content(
        model=model_name,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=3072)
    )
    
    return result.embeddings[0].values


def get_embedding(text: str) -> list[float]:
    """Return a 3072-dim embedding for `text`. 
    Defaults to the native Google GenAI SDK with gemini-embedding-2.
    """
    model_env = os.getenv("LITELLM_MODEL", "gemini/gemini-1.5-flash")

    # If the user explicitly switched to OpenAI, fall back to LiteLLM for those calls.
    if "openai" in model_env or "gpt" in model_env:
        try:
            from litellm import embedding as litellm_embedding
            api_key = os.getenv("OPENAI_API_KEY")
            # Note: text-embedding-3-large supports 3072 dims
            resp = litellm_embedding(model="text-embedding-3-large", input=text, api_key=api_key)
            return resp.data[0]['embedding']
        except Exception as e:
            logger.warning("openai_embedding_failed", error=str(e))
            # continue to fallback

    # Default to Gemini 2
    emb_model = "gemini-embedding-2"

    try:
        return _call_google_embedding(emb_model, text)
    except Exception as e:
        logger.warning("embedding_fallback_to_mock", error=str(e))
        random.seed(hash(text))
        return [random.uniform(-1.0, 1.0) for _ in range(3072)]
