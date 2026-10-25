
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

model_name = "gemini-3.5-flash-lite"

try:
    response = client.models.generate_content(
        model=model_name,
        contents="Say hello in one sentence."
    )

    print("Backup model works!")
    print(response.text)

except Exception as e:
    print("Backup model failed:")
    print(e)
