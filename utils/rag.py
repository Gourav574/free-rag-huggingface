import faiss
import numpy as np
from transformers import pipeline

qa_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

def build_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

def answer(query, chunks, index, embedder):
    q_emb = embedder([query])
    _, idx = index.search(np.array(q_emb), k=3)
    context = " ".join([chunks[i] for i in idx[0]])

    prompt = f"""
    Answer the question using the context below.
    Context: {context}
    Question: {query}
    """

    return qa_model(prompt)[0]["generated_text"]
