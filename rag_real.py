"""Real RAG: dense embeddings + similarity threshold + grounded LLM answer + citations + eval.

Reuses the hand-built pieces from rag_from_scratch.py (chunk_text, cosine_similarity, search).
Only the embedding step and the generation step are swapped for real models.

Run: uv run rag_real.py
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

from rag_from_scratch import chunk_text, search

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
EMBED_MODEL = "openai/text-embedding-3-small"
CHAT_MODEL = "openai/gpt-4o-mini"

NO_ANSWER = "I don't have enough information to answer that."

# ---------------------------------------------------------------------------
# WHY A THRESHOLD AND A "SAY I DON'T KNOW" INSTRUCTION
# Vector search ALWAYS returns top-k results, even when nothing is relevant
# (the TF-IDF demo returned a refund chunk for "capital of France").
# - Threshold (code-level guard): drop chunks whose cosine score is below MIN_SCORE.
#   If nothing survives, we return NO_ANSWER WITHOUT calling the LLM
#   -> cheaper, faster, and cannot hallucinate.
# - Prompt instruction (model-level guard): covers the case where chunks pass the
#   threshold but still don't contain the answer; the model must refuse, not guess.
# You need both: the threshold is a coarse filter, the instruction is the safety net.
# MIN_SCORE is NOT universal: it depends on the embedding model. Tune it on your
# own questions (look at the printed scores). For text-embedding-3-small,
# related ~0.4+, unrelated ~0.1-0.25.
# ---------------------------------------------------------------------------
MIN_SCORE = 0.30


def embed(texts):
    """Embed a list of strings in ONE API call -> list of vectors."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def build_cited_prompt(question, retrieved):
    """retrieved: list of (chunk_dict, score). Numbered sources enable citations."""
    context = "\n\n---\n\n".join(
        f"[Source {i + 1}] (from {c['source']})\n{c['text']}"
        for i, (c, _) in enumerate(retrieved)
    )
    return f"""Answer the question using ONLY the context below.
Cite the sources you used like [Source 1].
If the context does not contain the answer, reply exactly: "{NO_ANSWER}"

Context:
{context}

Question: {question}

Answer:"""


def chat(prompt, temperature=0):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,  # 0 = deterministic, best for factual RAG
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


class RealRAG:
    def __init__(self, min_score=MIN_SCORE):
        self.chunks = []  # [{"text":..., "source":...}]  (metadata enables citations)
        self.embeddings = []
        self.min_score = min_score

    def index(self, docs):
        """docs: {source_name: text}. Offline step: chunk -> embed -> store."""
        for source, text in docs.items():
            for piece in chunk_text(text, chunk_size=200, overlap=50):
                self.chunks.append({"text": piece, "source": source})
        self.embeddings = embed([c["text"] for c in self.chunks])

    def retrieve(self, question, top_k=3):
        """Embed query with the SAME model, brute-force cosine, apply threshold."""
        q_emb = embed([question])[0]
        hits = search(q_emb, self.embeddings, top_k)
        return [(self.chunks[i], score) for i, score in hits if score >= self.min_score]

    def answer(self, question, top_k=3):
        retrieved = self.retrieve(question, top_k)
        if not retrieved:  # threshold guard: skip the LLM entirely
            return NO_ANSWER, []
        return chat(build_cited_prompt(question, retrieved)), retrieved


# ---------------------------------------------------------------------------
# HOW TO MEASURE RAG (two separate questions: did we FIND it? did we USE it well?)
#
# 1) RETRIEVAL: recall@k
#    Golden set = questions + which source should be retrieved.
#    recall@k = (# questions whose correct source is in the top-k) / (# questions)
#    If recall@k is low, no prompt or LLM can fix it: fix chunking/embeddings/k.
#    (precision@k = fraction of the k retrieved chunks that are relevant.)
#
# 2) GENERATION: faithfulness
#    Is every claim in the answer supported by the retrieved context? (no
#    hallucination). Measured with an LLM-as-judge (below); RAGAS/DeepEval
#    automate this later. Also check answer relevance (does it address the
#    question?) and no-answer accuracy (refuses when it should).
# ---------------------------------------------------------------------------
def recall_at_k(rag, golden, k=3):
    hit = 0
    for question, expected_source in golden:
        # no threshold here: measure raw retrieval quality
        q_emb = embed([question])[0]
        top = search(q_emb, rag.embeddings, k)
        sources = {rag.chunks[i]["source"] for i, _ in top}
        hit += expected_source in sources
    return hit / len(golden)


def faithfulness(answer, retrieved):
    """LLM-as-judge: 1.0 if every claim is supported by the context, else 0.0."""
    context = "\n".join(c["text"] for c, _ in retrieved)
    verdict = chat(
        f"Context:\n{context}\n\nAnswer:\n{answer}\n\n"
        "Is every factual claim in the Answer supported by the Context? "
        "Reply with exactly YES or NO."
    )
    return 1.0 if verdict.upper().startswith("YES") else 0.0


if __name__ == "__main__":
    DOCS = {
        "refunds.md": "Our refund policy allows returns within 30 days of purchase. Refunds are issued to the original payment method within 5 business days.",
        "pricing.md": "The Pro plan costs 20 dollars per month and includes priority support. The Enterprise plan has custom pricing.",
        "account.md": "To reset your password, click Forgot Password on the login page and follow the email link.",
    }
    rag = RealRAG()
    rag.index(DOCS)

    print("=== Q&A ===")
    for q in [
        "How long do I have to return a product?",  # TF-IDF failed this one
        "How much is the Pro plan?",
        "What is the capital of France?",  # out of scope -> must refuse
    ]:
        ans, retrieved = rag.answer(q)
        print(f"Q: {q}")
        for c, s in retrieved:
            print(f"   {s:.3f} {c['source']}")
        print(f"A: {ans}")
        if retrieved:
            print(f"   faithfulness={faithfulness(ans, retrieved)}")
        print()

    print("=== Eval ===")
    GOLDEN = [
        ("How long do I have to return a product?", "refunds.md"),
        ("When will I get my money back?", "refunds.md"),
        ("How much does Pro cost?", "pricing.md"),
        ("Is there custom pricing?", "pricing.md"),
        ("I forgot my login credentials", "account.md"),
    ]
    for k in (1, 2):
        print(f"recall@{k} = {recall_at_k(rag, GOLDEN, k):.2f}")
