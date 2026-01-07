from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader


@dataclass
class PDFDoc:
    path: Path
    title: str | None
    pages: int
    texts: list[str]


def extract_pdf(path: Path) -> PDFDoc:
    reader = PdfReader(str(path))
    meta = reader.metadata or {}
    title = None
    try:
        title = getattr(meta, "title", None) or meta.get("/Title")
    except Exception:
        pass
    texts: list[str] = []
    for page in reader.pages:
        try:
            # extract_text() only extracts text, not images/graphics
            # Images would be accessed via page.images separately
            text = page.extract_text(extraction_mode="plain") or ""
        except Exception:
            text = ""
        texts.append(text.strip())
    return PDFDoc(path=path, title=title, pages=len(reader.pages), texts=texts)
