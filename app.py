from utils.loader import load_pdf
from utils.chunker import chunk_text
from utils.embeddings import embed_text
from utils.rag import build_faiss, answer

text = load_pdf("data/sample_policy.pdf")
chunks = chunk_text(text)

embeddings = embed_text(chunks)
index = build_faiss(embeddings)

while True:
    q = input("Ask a question (or 'exit'): ")
    if q.lower() == "exit":
        break
    print(answer(q, chunks, index, embed_text))
