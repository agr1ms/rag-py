# rag-py

RAG from first principles: raw embeddings → vector store → retrieval → grounded generation → evaluation, then the LangChain equivalent. Uses OpenRouter (OpenAI-compatible) for chat and embeddings.

## Run
```bash
cp .env.example .env   # add OPENROUTER_API_KEY
uv sync
uv run main.py                # smoke test
uv run raw_embeddings.py      # raw SDK chat + embedding
uv run vector_db.py           # Chroma upsert + query
uv run similarity_Search.py   # LangChain Chroma similarity search with scores
```

## Status
Done: embeddings, Chroma store, similarity search. Next: chunking, RAG prompt + generation with citations, eval set (recall@k). See the vault note `05-RAG`.
