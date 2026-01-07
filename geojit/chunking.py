from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class Chunk:
    index: int
    text: str
    page_start: int | None
    page_end: int | None
    token_count: int | None


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def chunk_pages(pages: Iterable[str], chunk_size: int = 1200, chunk_overlap: int = 200) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    idx = 0
    for pageno, text in enumerate(pages, start=1):
        for piece in _split_text(text, chunk_size, chunk_overlap):
            all_chunks.append(Chunk(index=idx, text=piece, page_start=pageno, page_end=pageno, token_count=None))
            idx += 1
    return all_chunks

