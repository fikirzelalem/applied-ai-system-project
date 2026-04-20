import os
from typing import List, Dict
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


FEW_SHOT_EXAMPLES = """
Example 1:
User: suggest something chill for studying
Cadence AI: For studying I'd go with Midnight Coding by LoRoom. It's lofi with a chill mood and energy of 0.42, which is low enough to stay in the background without pulling your attention. High acousticness at 0.71 gives it a warm, non-distracting feel. Library Rain by Paper Lanterns is another solid pick for the same reason.

Example 2:
User: I want something intense for the gym
Cadence AI: Storm Runner by Voltline is your best bet. Rock genre, intense mood, energy at 0.91 and tempo at 152 BPM. It's built for high output moments. If you want something more electronic, Bass Drop by Circuit Nine hits 0.96 energy at 140 BPM with an aggressive mood tag.

Example 3:
User: something moody and dark for a night drive
Cadence AI: Neon Jungle by Prism Wave fits perfectly. It's electronic with a moody feel, energy at 0.80, and tagged as dark. Night Drive Loop by Neon Echo is another option in the same space, synthwave with a moody vibe at 0.75 energy.
"""


def build_prompt(query: str, retrieved_docs: List[Dict]) -> str:
    """Build the prompt by combining few-shot examples and retrieved context with the user's query."""
    context = "\n\n".join(
        f"[{doc['source']}]\n{doc['content']}" for doc in retrieved_docs
    )
    return f"""You are Cadence AI, a music recommendation assistant with a direct, friendly tone.
Use only the information provided in the context below to answer the user's question.
Do not make up songs, artists, or facts that are not in the context.
Always sound like the examples below — specific, confident, and conversational. No bullet lists, just natural sentences.

--- FEW-SHOT EXAMPLES ---
{FEW_SHOT_EXAMPLES}
--- END EXAMPLES ---

--- CONTEXT ---
{context}
--- END CONTEXT ---

User question: {query}

Cadence AI:"""


def generate(query: str, retrieved_docs: List[Dict]) -> str:
    """Send the prompt to Gemini and return the response text."""
    prompt = build_prompt(query, retrieved_docs)
    try:
        response = client.models.generate_content(
            model="gemma-3-1b-it",
            contents=prompt
        )
        return response.text
    except Exception:
        sources = ", ".join(set(doc["source"] for doc in retrieved_docs))
        return (
            f"[Demo mode — Gemini API unavailable]\n\n"
            f"Based on your query, I found relevant information in: {sources}.\n"
            f"Once the API is connected, I'll generate a full recommendation here."
        )
