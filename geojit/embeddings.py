from __future__ import annotations

import os
from typing import Iterable
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt

try:
    from ai_sdk import embed_many, openai
    _ai_sdk_available = True
except Exception:  # pragma: no cover
    _ai_sdk_available = False


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def embed_texts(texts: list[str], api_key: str | None, model: str = "text-embedding-3-large") -> np.ndarray:
    if not texts:
        return np.zeros((0, 3072), dtype=np.float32)

    if not _ai_sdk_available:
        raise RuntimeError("ai-sdk-python not installed or import failed")

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY for embeddings")

    # Use ai_sdk's embed_many function
    embedding_model = openai.embedding(model)
    result = embed_many(model=embedding_model, values=texts)

    # Convert to numpy array
    vecs = [np.array(emb, dtype=np.float32) for emb in result.embeddings]
    return np.stack(vecs, axis=0)
