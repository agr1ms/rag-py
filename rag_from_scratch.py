import math
from collections import Counter

# Chunk 1 is words 0–199, chunk 2 is words 150–349, and so on.
# Moving by chunk_size - overlap makes consecutive chunks share 50 words. That way a sentence cut at a boundary still appears whole in one chunk.
# Short docs (like my demo) fit in one chunk, so chunking changes nothing there.
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Collects every unique lowercase word across all chunks into a sorted list, e.g. ["30", "days", "password", "refund", ...].
# Why: each position in a vector must mean the same word every time. The vocabulary fixes that order.
# The vector length equals the vocabulary size. This is why TF-IDF vectors are huge and mostly zeros, unlike the 1536 dense numbers from a real model.
def build_vocabulary(documents):
    vocab = set()
    for doc in documents:
        vocab.update(doc.lower().split())
    return sorted(vocab)

# For every vocabulary word, this gives the fraction of this text made of that word. A word that doesn’t appear gets 0. Result: a list with one number per vocabulary word.
def compute_tf(text, vocab):
    words = text.lower().split()
    count = Counter(words)
    total = len(words)
    if total == 0:
        return [0.0] * len(vocab)
    return [count.get(word, 0) / total for word in vocab]

# Words in every chunk (“the”, “is”) carry no information, so they get a low weight.
# A word in only one chunk (“refund”) identifies that chunk, so it gets a high weight.
# The +1 terms avoid dividing by zero and avoid a zero weight. This is the smoothed form that scikit-learn uses.
# The result is one weight per vocabulary word, computed once at indexing time.
def compute_idf(documents, vocab):
    n = len(documents)
    doc_sets = [set(doc.lower().split()) for doc in documents]  # split once, not per word
    idf = []
    for word in vocab:
        doc_count = sum(1 for words in doc_sets if word in words)
        idf.append(math.log((n + 1) / (doc_count + 1)) + 1)
    return idf

# tf × idf per word. The vector is large where a word is frequent in this text and rare across all chunks. That vector is the “embedding”.
def tfidf_embed(text, vocab, idf):
    tf = compute_tf(text, vocab)
    return [t * i for t, i in zip(tf, idf)]

# 1.0 means same direction (very similar). 0 means nothing in common.
# Dividing by the lengths means a long chunk isn’t favoured just for having bigger numbers.
def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

# Loop over every stored vector, score it against the query, sort descending, and keep the top k as (index, score) pairs. This is brute force: it compares the query to every vector. A real vector DB uses an approximate index (HNSW) instead of looping through all of them.
def search(query_embedding, stored_embeddings, top_k=5):
    scores = []
    for i, emb in enumerate(stored_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((i, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# It numbers each chunk as [Source 1], [Source 2], and so on. It joins them with separators and wraps them in an instruction: “answer ONLY from this context, else say you don’t have enough information”. The numbering enables citations, and the instruction is what produces grounded, no-hallucination answers.
def build_rag_prompt(query, retrieved_chunks):
    context = "\n\n---\n\n".join(
        f"[Source {i+1}]\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    )
    return f"""Answer the question based ONLY on the following context.
If the context doesn't contain enough information, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""



# index(documents) runs offline: chunk every doc, build the vocabulary, compute the IDF weights, then embed every chunk and store the vectors in self.embeddings.
# query(question, top_k) runs online: embed the question using the same vocabulary and IDF, search, look up the chunk text by index, and build the prompt. It returns the prompt and the scored chunks.
# Using the same vocab and IDF for the question is the “same embedding model for query and docs” rule in practice.
class RAGPipeline:
    def __init__(self):
        self.chunks = []
        self.embeddings = []
        self.vocab = []
        self.idf = []

    def index(self, documents):
        all_chunks = []
        for doc in documents:
            all_chunks.extend(chunk_text(doc))
        self.chunks = all_chunks
        self.vocab = build_vocabulary(all_chunks)
        self.idf = compute_idf(all_chunks, self.vocab)
        self.embeddings = [
            tfidf_embed(chunk, self.vocab, self.idf)
            for chunk in all_chunks
        ]

    def query(self, question, top_k=5):
        query_emb = tfidf_embed(question, self.vocab, self.idf)
        results = search(query_emb, self.embeddings, top_k)
        retrieved = [(self.chunks[i], score) for i, score in results]
        prompt = build_rag_prompt(
            question, [chunk for chunk, _ in retrieved]
        )
        return prompt, retrieved

# This is a fake LLM, only there so the pipeline has an end. It takes the words after "question:" in the prompt. It splits the chunks into sentences and returns the sentence sharing the most words with the question. A real version is one chat completion call with the prompt.
def simple_generate(prompt, retrieved_chunks):
    query_words = set(prompt.lower().split("question:")[-1].split())
    best_sentence = ""
    best_score = 0
    for chunk in retrieved_chunks:
        for sentence in chunk.split("."):
            sentence = sentence.strip()
            if not sentence:
                continue
            words = set(sentence.lower().split())
            overlap = len(query_words & words)
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
    return best_sentence if best_sentence else "I don't have enough information."

if __name__ == "__main__":
    docs = [
        "Our refund policy allows returns within 30 days of purchase. Refunds are issued to the original payment method within 5 business days.",
        "The Pro plan costs 20 dollars per month and includes priority support. The Enterprise plan has custom pricing.",
        "To reset your password, click Forgot Password on the login page and follow the email link.",
    ]
    rag = RAGPipeline()
    rag.index(docs)
    for q in ["How long do I have to return a product?", "What is the capital of France?"]:
        prompt, retrieved = rag.query(q, top_k=2)
        print(f"Q: {q}")
        for chunk, score in retrieved:
            print(f"  {score:.3f}  {chunk[:60]}...")
        print("A:", simple_generate(prompt, [c for c, _ in retrieved]), "\n")



# whole flow- docs → chunk → vocab → idf → embed each chunk → store        (index, once)
# question → embed (same vocab/idf) → cosine vs all → top-k → prompt → LLM   (query, per request)

# to remember -> TF-IDF only matches exact words, so this demo missed "return" vs "refund". 2. the real embedding model replaces steps 2-5 and nothing else changes
# 
# 
# 
# Result:
#  uv run rag_from_scratch.py

# Q: How long do I have to return a product?
#  0.199  To reset your password, click Forgot Password on the login p...
#  0.159  Our refund policy allows returns within 30 days of purchase....
# A: To reset your password, click Forgot Password on the login page and follow the email link
# 
# Observation:  For “How long do I have to return a product?”, the password-reset chunk ranked first (0.199) and the refund chunk second (0.159). The reason is that the question shares only words like “do”, “I” and “to” with the chunks, not “return” or “refund”. TF-IDF matches words, not meaning, so “return” and “refund” count as unrelated.

# Q: What is the capital of France?
#  0.243  Our refund policy allows returns within 30 days of purchase....
#  0.157  To reset your password, click Forgot Password on the login p...
# A: Our refund policy allows returns within 30 days of purchase
# 
# Observation: The unanswerable question, “What is the capital of France?”, got no “I don’t know”. The fake generator always returns the best-overlap sentence. The grounded prompt only produces an “I don’t know” with a real LLM, and a score threshold would also block it.
