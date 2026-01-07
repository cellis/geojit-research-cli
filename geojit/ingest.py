from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

from .config import load_settings
from .db import connect, ensure_schema, upsert_document, insert_chunks, upsert_company, insert_financial_metric
from .pdf_parser import extract_pdf
from .chunking import chunk_pages
from .embeddings import embed_texts
from .qdrant_store import get_qdrant, ensure_collection, upsert_points
from .metadata_extractor import extract_document_metadata


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def iter_pdfs(root: Path) -> list[Path]:
    pdfs: list[Path] = []
    for p in sorted(root.glob("**/*.pdf")):
        pdfs.append(p)
    return pdfs


def ingest_all(max_files: int | None = None) -> None:
    s = load_settings()
    conn = connect(s.database_url)
    ensure_schema(conn)

    pdfs = iter_pdfs(s.data_dir)
    if max_files is not None:
        pdfs = pdfs[:max_files]
    if not pdfs:
        print(f"No PDFs found under {s.data_dir}")
        return

    qdrant = get_qdrant(s.qdrant_url, s.qdrant_api_key)

    # Precreate collection once we know embedding size; we’ll lazily compute on the first batch
    vector_size = None

    for path in tqdm(pdfs, desc="Ingest PDFs"):
        try:
            sha = _sha256_file(path)
            doc = extract_pdf(path)
            doc_id = upsert_document(conn, str(path), sha, doc.title, doc.pages)
            chunks = chunk_pages(doc.texts, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)

            # Extract metadata from the full document text
            full_text = "\n\n".join(doc.texts)
            metadata = extract_document_metadata(
                full_text,
                path.name,
                api_key=s.openai_api_key,
                use_llm=True
            )

            # Store company and metrics if extracted
            if metadata.get("company_name"):
                try:
                    company_id = upsert_company(
                        conn,
                        name=metadata["company_name"],
                        sector=metadata.get("sector"),
                        industry=metadata.get("industry"),
                        metadata={"source": metadata.get("source", "unknown")}
                    )

                    # Store financial metrics if available
                    if "metrics" in metadata:
                        for metric in metadata["metrics"]:
                            insert_financial_metric(
                                conn,
                                company_id=company_id,
                                document_id=doc_id,
                                metric_name=metric["name"],
                                metric_value=metric.get("value"),
                                metric_value_text=metric["value_text"],
                                unit=metric.get("unit"),
                                period=metadata.get("period") or metric.get("period"),
                                quarter=metadata.get("quarter"),
                                fiscal_year=metadata.get("fiscal_year"),
                                metadata={}
                            )
                except Exception as meta_err:
                    print(f"  Warning: Failed to store metadata for {path.name}: {meta_err}")

            # Insert to Postgres first
            insert_chunks(
                conn,
                document_id=doc_id,
                chunks=[(c.index, c.text, c.page_start, c.page_end, c.token_count) for c in chunks],
            )

            # Embed and send to Qdrant in moderately sized batches for memory
            batch = 64
            for i in range(0, len(chunks), batch):
                slice_chunks = chunks[i : i + batch]
                texts = [c.text for c in slice_chunks]
                vecs = embed_texts(texts, api_key=s.openai_api_key, model=s.embedding_model)
                if vector_size is None:
                    vector_size = vecs.shape[1]
                    ensure_collection(qdrant, s.qdrant_collection, vector_size)
                ids = [f"{doc_id}:{c.index}" for c in slice_chunks]
                payloads = [
                    {
                        "document_id": doc_id,
                        "path": str(path),
                        "chunk_index": c.index,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        "title": doc.title,
                        "text": c.text,
                    }
                    for c in slice_chunks
                ]
                upsert_points(qdrant, s.qdrant_collection, ids, vecs.tolist(), payloads)
        except Exception as e:
            print(f"Failed to ingest {path}: {e}")
