import os
import json
from datetime import datetime


LOG_FILE = "logs/interactions.jsonl"


def log(query: str, retrieved_docs: list, response: str, guardrail_passed: bool) -> None:
    """Append a single interaction to the log file as a JSON line."""
    os.makedirs("logs", exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "guardrail_passed": guardrail_passed,
        "retrieved_sources": [doc["source"] for doc in retrieved_docs],
        "response": response,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
