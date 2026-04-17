import time
from sentence_transformers import CrossEncoder

# Load model once
reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, chunks: list, score_threshold: float ):
    start = time.time()

    # Create (query, chunk) pairs
    pairs = [(query, chunk) for chunk in chunks]

    # Get relevance scores
    scores = reranker_model.predict(pairs)

    # Combine chunks with scores
    scored_chunks = list(zip(chunks, scores))

    # Sort by score (descending)
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # ✅ Filter based on threshold
    filtered_chunks = [
        chunk for chunk, score in scored_chunks if score >= score_threshold
    ]

    # Fallback (important)
    if len(filtered_chunks) == 0:
        filtered_chunks = [scored_chunks[0][0]]

    elapsed = time.time() - start

    return filtered_chunks, elapsed