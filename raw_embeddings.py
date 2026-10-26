import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

conversation = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role":"system", "content":"here's agrim's AI"},
        {"role":"user", "content":"what is starship?"}
    ],
)

print(conversation.choices[0].message.content)

response = client.embeddings.create(
    input="Here goes the string", model="openai/text-embedding-3-small"
)

print(response)

# see, here I will wrap the embedding response in len to get the size of embedding
# why? becoz this size gives us the embedding dimensions. as the numbers can be compared easily, we can cal distance btwn 2 vectors, find which vectors are similar.
# diff models give diff size dimensions. more dimensions -> more meanings, more features of whatever we have trasnsfered into the vector.
print(len(response.data[0].embedding))