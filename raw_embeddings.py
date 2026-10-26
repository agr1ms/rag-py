import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

response = client.embeddings.create(
    input="Here goes the string", model="openai/text-embedding-3-small"
)

print(response)

conversation = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"system", "content":"here's agrim's AI"},
        {"role":"user", "content":"what is starship?"}
    ],
)

print(conversation.choices[0].message.content)