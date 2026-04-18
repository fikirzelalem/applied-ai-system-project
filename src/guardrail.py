from typing import List, Dict


BLOCKED_TOPICS = ["politics", "religion", "violence", "drugs", "hate"]


def is_off_topic(query: str) -> bool:
    """Return True if the query contains blocked topics unrelated to music."""
    query_lower = query.lower()
    return any(topic in query_lower for topic in BLOCKED_TOPICS)


def has_enough_context(retrieved_docs: List[Dict]) -> bool:
    """Return True if retrieval found at least one relevant document."""
    return len(retrieved_docs) > 0


def check(query: str, retrieved_docs: List[Dict]) -> tuple[bool, str]:
    """
    Run all guardrail checks. Returns (passed, reason).
    If passed is False, reason explains why the system should not respond.
    """
    if is_off_topic(query):
        return False, "I can only help with music recommendations. That topic is outside my scope."

    if not has_enough_context(retrieved_docs):
        return False, "I don't have enough information to answer that confidently. Try asking about a genre, mood, or artist in the catalog."

    return True, ""
