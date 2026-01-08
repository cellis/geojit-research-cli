from __future__ import annotations

import numpy as np
from typing import Sequence

from .config import load_settings
from .embeddings import embed_texts
from .qdrant_store import get_qdrant, search

try:
    import fitz  # PyMuPDF
    _fitz_available = True
except Exception:
    _fitz_available = False

try:
    from pypdf import PdfReader
    _pypdf_available = True
except Exception:
    _pypdf_available = False


def _extract_page_text(pdf_path: str, page_start: int | None, page_end: int | None, limit_chars: int = 1200) -> str:
    """Best-effort page text extraction for citations when Qdrant payload omits text."""
    if not pdf_path:
        return ""
    try:
        if _fitz_available:
            doc = fitz.open(pdf_path)
            try:
                a = max(1, page_start or 1)
                b = min(len(doc), page_end or a)
                out = []
                for i in range(a - 1, b):
                    try:
                        out.append(doc.load_page(i).get_text("text") or "")
                    except Exception:
                        pass
                text = "\n".join(out).strip()
                return text[:limit_chars]
            finally:
                doc.close()
        elif _pypdf_available:
            reader = PdfReader(pdf_path)
            a = max(1, page_start or 1)
            b = min(len(reader.pages), page_end or a)
            out = []
            for i in range(a - 1, b):
                try:
                    out.append(reader.pages[i].extract_text() or "")
                except Exception:
                    pass
            return ("\n".join(out).strip())[:limit_chars]
    except Exception:
        return ""
    return ""


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    s = load_settings()
    k = top_k or s.top_k
    qvec = embed_texts([query], api_key=s.openai_api_key, model=s.embedding_model)[0]
    client = get_qdrant(s.qdrant_url, s.qdrant_api_key)
    hits = search(client, s.qdrant_collection, qvec.tolist(), top_k=k)
    results: list[dict] = []
    for h in hits:
        p = h.payload or {}
        item = {
            "score": float(h.score),
            "text": p.get("text"),  # may be missing; we fetch from PDF below if needed
            "path": p.get("path"),
            "document_id": p.get("document_id"),
            "chunk_index": p.get("chunk_index"),
            "page_start": p.get("page_start"),
            "page_end": p.get("page_end"),
            "title": p.get("title"),
            "id": h.id,
        }
        if not item["text"]:
            item["text"] = _extract_page_text(item.get("path"), item.get("page_start"), item.get("page_end"))
        results.append(item)
    return results


def retrieve_company(query: str, top_k: int | None = None) -> list[dict]:
    """Search the dedicated companies collection by name to get nearest companies.

    Returns a list of payload dicts including company/organization ids when available.
    Returns empty list if collection doesn't exist yet.
    """
    s = load_settings()
    k = top_k or s.top_k
    qvec = embed_texts([query], api_key=s.openai_api_key, model=s.embedding_model)[0]
    client = get_qdrant(s.qdrant_url, s.qdrant_api_key)
    collection = f"{s.qdrant_collection}-companies"

    # Check if collection exists first
    try:
        client.get_collection(collection)
    except Exception:
        # Collection doesn't exist yet (no companies ingested)
        return []

    try:
        hits = search(client, collection, qvec.tolist(), top_k=k)
    except Exception:
        # Search failed for some reason
        return []

    results: list[dict] = []
    for h in hits:
        p = h.payload or {}
        results.append({
            "score": float(h.score),
            "name": p.get("name"),
            "company_id": p.get("company_id"),
            "organization_id": p.get("organization_id"),
            "sector": p.get("sector"),
            "industry": p.get("industry"),
            "id": h.id,
        })
    return results
