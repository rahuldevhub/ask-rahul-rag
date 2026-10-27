"""Resolve short follow-ups from cited topics without extra LLM calls."""

import re

MAX_MESSAGES = 20
FOLLOW_UP = re.compile(
    r"\b(?:that|this)\s+(?:project|role|job|work|app|platform)\b|"
    r"\b(?:it|those|these)\b|^(?:and\b|tell me more\b|what else\b)|"
    r"\b(?:what|which)\s+(?:tools|technologies|tech stack)\s+did\s+(?:he|rahul)\s+use\b",
    re.IGNORECASE,
)


def resolve_context(question: str, history: list[dict], known_topics=()) -> tuple[str, str]:
    """Return (context, clarification). Explicit topics override prior context."""
    if any(topic.casefold() in question.casefold() for topic in known_topics if len(topic) >= 4):
        return "", ""
    if not FOLLOW_UP.search(question):
        return "", ""
    last = history[-1] if history else {}
    if last.get("role") != "assistant" or last.get("status") != "answered":
        return "", "Which project or role do you mean? Please include its name."
    topics = list(dict.fromkeys(source["label"] for source in last.get("sources", [])))
    if len(topics) != 1:
        return "", "Which project or role do you mean? The previous answer covered more than one topic."
    return topics[0][:200], ""


def submit_question(question: str, history: list[dict], assistant) -> dict:
    known_topics = [chunk["label"].split(": ", 1)[-1].split(" | ", 1)[0]
                    for chunk in assistant.manifest["chunks"] if ": " in chunk["label"]]
    context, clarification = resolve_context(question, history, known_topics)
    if clarification:
        return {"status": "clarification", "answer": clarification, "sources": []}
    return assistant.ask(question, conversation_context=context)
