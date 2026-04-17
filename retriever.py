import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2") 




# 1. Load TXT file

with open("data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()   



# 2. Chunking function

def chunk_text(text, chunk_size=200, overlap=30):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = end - overlap

        if start >= len(words):
            break

    return chunks


texts = chunk_text(raw_text)



# 3. Embeddings

embeddings = model.encode(texts)
embeddings = np.array(embeddings).astype("float32")



# 4. FAISS index

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)



# 5. Retrieve function

def retrieve(query: str, top_k: int = 5):
    start = time.time()

    query_vec = model.encode([query])
    query_vec = np.array(query_vec).astype("float32")

    distances, indices = index.search(query_vec, top_k)

    results = [texts[i] for i in indices[0]]

    elapsed = time.time() - start

    return results, elapsed

