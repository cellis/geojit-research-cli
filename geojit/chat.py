from __future__ import annotations

from collections import deque
from typing import Deque

from .config import load_settings
from .retriever import retrieve

try:
    from ai_sdk import generate_text, openai
    _ai_sdk_available = True
except Exception:  # pragma: no cover
    _ai_sdk_available = False


class ChatSession:
    def __init__(self, max_turns: int = 20):
        self.history: Deque[dict] = deque(maxlen=max_turns * 2)

    def add_user(self, content: str):
        self.history.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.history.append({"role": "assistant", "content": content})


def _rewrite_query(query: str, session: ChatSession | None) -> str:
    # Lightweight heuristic: if pronouns like "that/it" appear and we have previous user messages,
    # append the last user query for extra context to retrieval.
    if not session or not session.history:
        return query
    qlow = query.lower()
    if any(p in qlow for p in ["that ", "it ", "they ", "last quarter", "last year"]):
        prev = [m["content"] for m in session.history if m.get("role") == "user"]
        if prev:
            return f"{query} (context: {prev[-1]})"
    return query


def answer(query: str, session: ChatSession | None = None, deep_research: bool = False) -> str:
    s = load_settings()
    if not _ai_sdk_available:
        raise RuntimeError("ai-sdk-python not available")

    from ai_sdk.types import CoreSystemMessage, CoreUserMessage, CoreAssistantMessage, TextPart
    from datetime import datetime

    # Check if user is requesting deep research
    query_lower = query.lower()
    deep_triggers = ['deep research', 'think hard', 'analyze thoroughly', 'detailed analysis', 'investigate']
    if not deep_research:
        deep_research = any(trigger in query_lower for trigger in deep_triggers)

    # Retrieve context
    r_query = _rewrite_query(query, session)
    contexts = retrieve(r_query, top_k=s.top_k)
    context_texts = []
    for c in contexts:
        title = c.get("title") or ""
        loc = f"p.{c.get('page_start')}"
        ctx = c.get("text") or ""
        context_texts.append(f"[{title} {loc}]\n{ctx}")
    context_blob = "\n\n---\n\n".join(context_texts)

    # Use deep research mode if requested
    if deep_research:
        from .coding_agent import deep_research as run_deep_research
        return run_deep_research(query, context=context_blob)

    # Get current date/time for temporal grounding
    now = datetime.now()
    date_context = f"Today is {now.strftime('%B %d, %Y')} ({now.strftime('%A')}). Current time: {now.strftime('%I:%M %p')}."

    sys_prompt = (
        f"{date_context}\n\n"
        "You are a financial research assistant. Answer only using the provided context from PDFs. "
        "Cite using [Title p.X] for each key claim. If information is missing in context, say you don't have it. "
        "When users ask about 'last year', 'last quarter', etc., use today's date to determine the relevant time period."
    )

    # Build messages using ai_sdk types
    messages = [CoreSystemMessage(content=sys_prompt)]
    if session:
        for msg in session.history:
            if msg["role"] == "user":
                messages.append(CoreUserMessage(content=[TextPart(text=msg["content"])]))
            elif msg["role"] == "assistant":
                messages.append(CoreAssistantMessage(content=[TextPart(text=msg["content"])]))

    # Add current query with context
    user_content = f"Context:\n{context_blob}\n\nQuestion: {query}"
    messages.append(CoreUserMessage(content=[TextPart(text=user_content)]))

    # Generate response using ai_sdk
    model = openai(s.openai_model)
    resp = generate_text(model=model, messages=messages)
    text = resp.text

    if session:
        session.add_user(query)
        session.add_assistant(text)
    return text
