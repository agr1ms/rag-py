import os

from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI

print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")

def main():
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        temperature=0,
    )

    # testing my api key
    response = llm.invoke("Say 'setup complete' in one word!")
    print(response.content)
    print(f"Response from llm: {response}")

    print(f"setup complete")

if __name__ == "__main__":
    main()
