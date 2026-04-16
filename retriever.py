import time
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load data
with open("chunks.json") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

# Create embeddings
embeddings = model.encode(texts)

# FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype("float32"))

def retrieve(query: str, top_k: int = 5):
    start = time.time()

    query_vec = model.encode([query])
    query_vec = np.array(query_vec).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = [texts[i] for i in indices[0]]

    elapsed = time.time() - start

    return results, elapsed