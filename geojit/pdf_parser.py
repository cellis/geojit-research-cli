from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class PDFDoc:
    path: Path
    title: str | None
    pages: int
    texts: list[str]


def extract_pdf(path: Path) -> PDFDoc:
    """Extract text and metadata from a PDF quickly.

    Prefers PyMuPDF (fitz) for speed and robustness. Falls back to pypdf if
    PyMuPDF is unavailable.
    """
    if _fitz_available:
        doc = fitz.open(str(path))
        try:
            meta = doc.metadata or {}
            title = meta.get("title") or meta.get("Title")
            texts: list[str] = []
            for i in range(len(doc)):
                try:
                    page = doc.load_page(i)
                    text = page.get_text("text") or ""
                except Exception:
                    text = ""
                texts.append(text.strip())
            return PDFDoc(path=path, title=title, pages=len(doc), texts=texts)
        finally:
            doc.close()

    if not _pypdf_available:
        # No backend available
        return PDFDoc(path=path, title=None, pages=0, texts=[])

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
            text = page.extract_text(extraction_mode="plain") or ""
        except Exception:
            text = ""
        texts.append(text.strip())
    return PDFDoc(path=path, title=title, pages=len(reader.pages), texts=texts)
