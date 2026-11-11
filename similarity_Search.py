import os
import tempfile

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings

load_dotenv()

# Embeddings model: turns text into a vector (list of floats) capturing meaning.
# check_embedding_ctx_length=False is needed for OpenRouter: langchain would otherwise
# send token IDs instead of raw text, which non-OpenAI endpoints don't accept.
embeddings_model = OpenAIEmbeddings(
    model="openai/text-embedding-3-small",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    check_embedding_ctx_length=False,
)

SAMPLE_DOCS = [
    Document(
        page_content="A vector store keeps text as embeddings so you can search by meaning.",
        metadata={"source": "vector_stores.md"},
    ),
    Document(
        page_content="Chroma is an open-source vector database that runs locally.",
        metadata={"source": "chroma.md"},
    ),
    Document(
        page_content="Embeddings map text to high-dimensional vectors; similar text lands close together.",
        metadata={"source": "embeddings.md"},
    ),
    Document(
        page_content="Paris is the capital of France and is known for the Eiffel Tower.",
        metadata={"source": "geography.md"},
    ),
    Document(
        page_content="Sourdough bread needs flour, water, salt and a fermented starter.",
        metadata={"source": "cooking.md"},
    ),
]


def similarity_search_with_scores():
    with tempfile.TemporaryDirectory() as tmpdir:
        # embeds every doc and stores (vector, text, metadata) in Chroma
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        # embeds the query, then finds the k nearest stored vectors
        query = "Explain vector stores."
        results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

        print(f"Top 3 results with scores for query '{query}':")
        for i, (doc, score) in enumerate(results_with_scores):
            # Chroma returns a DISTANCE (lower = closer), so flip it to a similarity
            final_score = 1 / (1 + score)
            print(
                f"Result {i+1}: {doc.page_content} (Score: {final_score:.4f}, Source: {doc.metadata['source']})"
            )


if __name__ == "__main__":
    similarity_search_with_scores()
