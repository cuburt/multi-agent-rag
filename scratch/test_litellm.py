import os
import litellm
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("LITELLM_MODEL")
print(f"Testing model: {model}")

try:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.0
    )
    print("Success!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
