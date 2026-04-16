import time
import google.generativeai as genai
import os
from dotenv import load_dotenv     

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-3-flash-preview")

def build_context(chunks):
    return "\n\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])

def generate_answer(query: str, chunks: list):
    context = build_context(chunks)

    prompt = f"""
You are a helpful assistant.
Answer ONLY from the given context.

Context:
{context}

Question: {query}
"""

    start = time.time()

    response = model.generate_content(prompt)

    elapsed = time.time() - start

    answer = response.text

    # Gemini doesn't always return tokens directly
    # tokens = getattr(response, "usage_metadata", {}).get("total_token_count", "N/A")

    return answer, elapsed