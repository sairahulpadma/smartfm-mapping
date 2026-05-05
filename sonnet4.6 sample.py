import os
import httpx
from dotenv import load_dotenv
from anthropic import AnthropicFoundry

load_dotenv()

endpoint        = os.getenv("AZURE_ANTHROPIC_ENDPOINT", "https://admv-mogidbp0-eastus2.services.ai.azure.com/anthropic/")
deployment_name = os.getenv("AZURE_ANTHROPIC_DEPLOYMENT", "claude-sonnet-4-6")
api_key         = os.getenv("AZURE_ANTHROPIC_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")

if not api_key:
    raise EnvironmentError("AZURE_ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill in your key.")

client = AnthropicFoundry(
    api_key=api_key,
    base_url=endpoint,
    http_client=httpx.Client(verify=False)
)

message = client.messages.create(
    model=deployment_name,
    messages=[
        {"role": "user", "content": "What do I do to make a simple python http server?"}
    ],
    max_tokens=1024,
)

print(message.content)