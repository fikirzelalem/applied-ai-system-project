import os
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")


def build_prompt(query: str, retrieved_docs: List[Dict]) -> str:
    """Build the prompt by combining retrieved context with the user's query."""
    context = "\n\n".join(
        f"[{doc['source']}]\n{doc['content']}" for doc in retrieved_docs
    )
    return f"""You are MoodMix AI, a music recommendation assistant.
Use only the information provided below to answer the user's question.
Do not make up songs, artists, or facts that are not in the context.

--- CONTEXT ---
{context}
--- END CONTEXT ---

User question: {query}

Give a helpful, conversational recommendation based strictly on the context above."""


def generate(query: str, retrieved_docs: List[Dict]) -> str:
    """Send the prompt to Gemini and return the response text."""
    prompt = build_prompt(query, retrieved_docs)
    response = model.generate_content(prompt)
    return response.text
