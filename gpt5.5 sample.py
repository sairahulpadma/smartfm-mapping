import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint    = os.getenv("AZURE_OPENAI_ENDPOINT", "https://admv-mogidbp0-eastus2.cognitiveservices.azure.com/")
model_name  = os.getenv("AZURE_OPENAI_MODEL",    "gpt-5.5")
deployment  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.5_1")
api_key     = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

if not api_key:
    raise EnvironmentError("AZURE_OPENAI_API_KEY is not set. Copy .env.example to .env and fill in your key.")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_completion_tokens=16384,
    model=deployment
)

print(response.choices[0].message.content)