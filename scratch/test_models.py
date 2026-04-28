import os
import litellm
from dotenv import load_dotenv

load_dotenv()

models_to_test = [
    "openrouter/meta-llama/llama-3.1-8b-instruct:free",
    "openrouter/meta-llama/llama-3-8b-instruct:free",
    "openrouter/google/gemma-2-9b-it:free"
]

for model in models_to_test:
    print(f"\n--- Testing model: {model} ---")
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.0
        )
        print("Success!")
        print(f"Response: {response.choices[0].message.content}")
        break
    except Exception as e:
        print(f"Error for {model}: {e}")
