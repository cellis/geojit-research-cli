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


def answer(query: str, session: ChatSession | None = None) -> str:
    s = load_settings()
    if not _ai_sdk_available:
        raise RuntimeError("ai-sdk-python not available")

    from ai_sdk.types import CoreSystemMessage, CoreUserMessage, CoreAssistantMessage, TextPart

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

    sys_prompt = (
        "You are a financial research assistant. Answer only using the provided context from PDFs. "
        "Cite using [Title p.X] for each key claim. If information is missing in context, say you don't have it."
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
