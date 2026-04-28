import os
from dotenv import load_dotenv

# Try to load .env manually to see if it makes a difference
load_dotenv()

print(f"LITELLM_MODEL: {os.getenv('LITELLM_MODEL')}")
print(f"OPENROUTER_API_KEY: {os.getenv('OPENROUTER_API_KEY')[:10]}...")
print(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')[:10]}...")
