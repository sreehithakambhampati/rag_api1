from fastapi import FastAPI
from pydantic import BaseModel
from retriever import retrieve
from rag_chain import generate_answer
from logger import logger

print("THIS IS MY MAIN FILE")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

print("ASK ROUTE LOADED")

@app.get("/")
def home():
    return {"message": "RAG API is running"}

print("ASK ROUTE LOADED")

@app.post("/ask")
def ask(request: QueryRequest):
    query = request.query 

    # Step 1: Retrieval
    chunks, retrieval_time = retrieve(query, top_k=5)
    logger.info(f"Retrieval time: {retrieval_time:.3f}s")

    # Step 2: Generation
    answer, llm_time, tokens = generate_answer(query, chunks)
    logger.info(f"LLM time: {llm_time:.3f}s | Tokens: {tokens}")

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": chunks,
        "logs": {
            "retrieval_time_sec": round(retrieval_time, 3),
            "llm_time_sec": round(llm_time, 3),
            "tokens_used": tokens
        }
    }