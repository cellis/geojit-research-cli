from __future__ import annotations

import numpy as np
from typing import Sequence

from .config import load_settings
from .embeddings import embed_texts
from .qdrant_store import get_qdrant, search


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    s = load_settings()
    k = top_k or s.top_k
    qvec = embed_texts([query], api_key=s.openai_api_key, model=s.embedding_model)[0]
    client = get_qdrant(s.qdrant_url, s.qdrant_api_key)
    hits = search(client, s.qdrant_collection, qvec.tolist(), top_k=k)
    results: list[dict] = []
    for h in hits:
        p = h.payload or {}
        results.append({
            "score": float(h.score),
            "text": p.get("text"),  # may be missing unless we store text; we didn’t to save space
            "path": p.get("path"),
            "document_id": p.get("document_id"),
            "chunk_index": p.get("chunk_index"),
            "page_start": p.get("page_start"),
            "page_end": p.get("page_end"),
            "title": p.get("title"),
            "id": h.id,
        })
    return results

