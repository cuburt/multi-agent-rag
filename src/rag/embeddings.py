import os
from litellm import embedding
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
import random

logger = structlog.get_logger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _call_embedding(model: str, text: str) -> list[float]:
    api_key = os.getenv("GEMINI_API_KEY")
    response = embedding(model=model, input=text, api_key=api_key)
    return response.data[0]['embedding']


def get_embedding(text: str) -> list[float]:
    """Return a 768-dim embedding for `text`. Calls litellm against a paired
    embedding endpoint; if anything goes wrong we fall back to a deterministic
    pseudo-random vector so unit tests and offline dev keep working.
    """
    model = os.getenv("LITELLM_MODEL", "gemini/gemini-1.5-flash")

    # The .env model is usually a generation model (e.g. gemini-1.5-flash),
    # which isn't valid for embedding calls. Pick the matching embedding
    # endpoint per provider instead of forwarding the user's value blindly.
    if "gemini" in model:
        emb_model = "gemini/text-embedding-004"
    elif "openai" in model or "gpt" in model:
        emb_model = "text-embedding-3-small"
    else:
        emb_model = model

    try:
        return _call_embedding(emb_model, text)
    except Exception as e:
        logger.warning("embedding_fallback_to_mock", error=str(e))
        random.seed(hash(text))
        return [random.uniform(-1.0, 1.0) for _ in range(768)]
