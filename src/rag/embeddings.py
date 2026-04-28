import os
from litellm import embedding
import random

def get_embedding(text: str) -> list[float]:
    """
    Get vector embedding for a given text.
    Uses litellm if a model is configured, otherwise falls back to a deterministic mock embedding for testing.
    """
    model = os.getenv("LITELLM_MODEL", "gemini/gemini-1.5-flash")
    
    try:
        # litellm requires the specific embedding endpoint models, like 'gemini/text-embedding-004'
        # if the user provided gemini-1.5-flash, that's for text generation. 
        # For embeddings, we should use a proper embedding model.
        if "gemini" in model:
            emb_model = "gemini/text-embedding-004"
        elif "openai" in model or "gpt" in model:
            emb_model = "text-embedding-3-small"
        else:
            emb_model = model # fallback
            
        response = embedding(model=emb_model, input=text)
        return response.data[0]['embedding']
    except Exception as e:
        print(f"Warning: Using mock embeddings due to error: {e}")
        # Deterministic pseudo-random mock embedding (length 768 to match our DB schema)
        random.seed(hash(text))
        return [random.uniform(-1.0, 1.0) for _ in range(768)]
