from __future__ import annotations

import os
from typing import Iterable
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt
import hashlib
import json
from pathlib import Path
from typing import Iterable

try:
    from ai_sdk import embed_many, openai
    _ai_sdk_available = True
except Exception:  # pragma: no cover
    _ai_sdk_available = False


_CACHE_DIR = Path(".cache/embeddings")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_get(h: str) -> np.ndarray | None:
    p = _CACHE_DIR / f"{h}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        arr = np.array(data["vec"], dtype=np.float32)
        return arr
    except Exception:
        return None


def _cache_put(h: str, vec: np.ndarray) -> None:
    try:
        p = _CACHE_DIR / f"{h}.json"
        p.write_text(json.dumps({"vec": vec.tolist()}), encoding="utf-8")
    except Exception:
        pass


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(5))
def embed_texts(texts: list[str], api_key: str | None, model: str = "text-embedding-3-large") -> np.ndarray:
    if not texts:
        return np.zeros((0, 3072), dtype=np.float32)

    if not _ai_sdk_available:
        raise RuntimeError("ai-sdk-python not installed or import failed")

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY for embeddings")

    # Resolve caching for each input text
    hashes = [_hash_text(t) for t in texts]
    cached: list[np.ndarray | None] = [_cache_get(h) for h in hashes]
    to_embed_indices = [i for i, v in enumerate(cached) if v is None]

    # Embed only uncached
    if to_embed_indices:
        values = [texts[i] for i in to_embed_indices]
        embedding_model = openai.embedding(model)
        result = embed_many(model=embedding_model, values=values)
        new_vecs = [np.array(emb, dtype=np.float32) for emb in result.embeddings]
        # Persist
        for idx, vec in zip(to_embed_indices, new_vecs):
            _cache_put(hashes[idx], vec)
            cached[idx] = vec

    # Assemble final array in original order
    final_vecs = [c if c is not None else np.zeros((3072,), dtype=np.float32) for c in cached]
    return np.stack(final_vecs, axis=0)
