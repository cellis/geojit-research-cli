#!/usr/bin/env python3
"""
Ingest a single PDF into the geojit-test database for testing.
"""
import sys
from pathlib import Path
from geojit.config import load_settings
from geojit.db import connect, ensure_schema, upsert_document, insert_chunks, upsert_company, upsert_organization
from geojit.pdf_parser_text import parse_pdf_text_based
from geojit.pdf_parser import extract_pdf
from geojit.chunking import chunk_pages
from geojit.embeddings import embed_texts
from geojit.qdrant_store import get_qdrant, ensure_collection, upsert_points
import hashlib

DATABASE_URL = "postgresql://localhost/geojit-test"
PDF_PATH = "./Financial_Research_Agent_Files/SP20241406115209223TTK.pdf"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _insert_profit_loss_quarterly(conn, org_id: str, data: dict) -> None:
    """Insert quarterly P&L data."""
    fields = {k: v for k, v in data.items() if k not in ['period', 'quarter', 'fiscal_year'] and v is not None}
    if not fields:
        return

    columns = ', '.join(['organization_id', 'period', 'quarter', 'fiscal_year'] + list(fields.keys()))
    placeholders = ', '.join(['%s'] * (4 + len(fields)))
    sql = f"INSERT INTO profit_loss_quarterly({columns}) VALUES ({placeholders})"

    with conn.cursor() as cur:
        cur.execute(sql, (org_id, data.get('period'), data.get('quarter'), data.get('fiscal_year'), *fields.values()))
    conn.commit()


def _insert_profit_loss(conn, org_id: str, data: dict) -> None:
    """Insert annual P&L data."""
    fields = {k: v for k, v in data.items() if k != 'fiscal_year' and v is not None}
    if not fields:
        return

    columns = ', '.join(['organization_id', 'fiscal_year'] + list(fields.keys()))
    placeholders = ', '.join(['%s'] * (2 + len(fields)))
    sql = f"INSERT INTO profit_loss({columns}) VALUES ({placeholders})"

    with conn.cursor() as cur:
        cur.execute(sql, (org_id, data.get('fiscal_year'), *fields.values()))
    conn.commit()


def _insert_balance_sheet(conn, org_id: str, data: dict) -> None:
    """Insert balance sheet data."""
    fields = {k: v for k, v in data.items() if k != 'fiscal_year' and v is not None}
    if not fields:
        return

    columns = ', '.join(['organization_id', 'fiscal_year'] + list(fields.keys()))
    placeholders = ', '.join(['%s'] * (2 + len(fields)))
    sql = f"INSERT INTO balance_sheet({columns}) VALUES ({placeholders})"

    with conn.cursor() as cur:
        cur.execute(sql, (org_id, data.get('fiscal_year'), *fields.values()))
    conn.commit()


def _insert_change_in_estimate(conn, org_id: str, data: dict) -> None:
    """Insert change in estimate data."""
    fields = {k: v for k, v in data.items() if k != 'fiscal_year' and v is not None}
    if not fields:
        return

    columns = ', '.join(['organization_id', 'fiscal_year'] + list(fields.keys()))
    placeholders = ', '.join(['%s'] * (2 + len(fields)))
    sql = f"INSERT INTO change_in_estimate({columns}) VALUES ({placeholders})"

    with conn.cursor() as cur:
        cur.execute(sql, (org_id, data.get('fiscal_year'), *fields.values()))
    conn.commit()


def ingest_pdf():
    """Ingest the TTK PDF into test database."""
    print(f"Ingesting {PDF_PATH} into {DATABASE_URL}...")

    # Load settings for API keys
    s = load_settings()

    # Connect to test database
    conn = connect(DATABASE_URL)
    ensure_schema(conn)

    path = Path(PDF_PATH)
    sha = _sha256_file(path)

    # Extract PDF
    print("  Extracting PDF...")
    doc = extract_pdf(path)
    doc_id = upsert_document(conn, str(path), sha, doc.title, doc.pages)
    print(f"  ✓ Document ID: {doc_id}")

    # Parse with text-based parser
    print("  Parsing PDF for structured data...")
    company_hint = "TTK"
    parsed_data = parse_pdf_text_based(str(path), company_hint=company_hint)

    # Create company and organization
    company_name = parsed_data.get('company_name', company_hint)
    print(f"  Company: {company_name}")

    company_id = upsert_company(
        conn,
        name=company_name,
        sector=parsed_data.get('sector'),
        industry=parsed_data.get('industry'),
        metadata={}
    )

    org_id = upsert_organization(
        conn,
        name=company_name,
        company_id=company_id,
        document_id=doc_id,
        sector=parsed_data.get('sector'),
        industry=parsed_data.get('industry'),
        pdf_path=str(path),
        parsing_method='text',
        metadata={}
    )
    print(f"  ✓ Organization ID: {org_id}")

    # Insert financial data
    print("  Inserting financial data...")

    qpl_count = 0
    for item in parsed_data.get('profit_loss_quarterly', []):
        _insert_profit_loss_quarterly(conn, org_id, item)
        qpl_count += 1
    print(f"    - Quarterly P&L: {qpl_count} records")

    pl_count = 0
    for item in parsed_data.get('profit_loss', []):
        _insert_profit_loss(conn, org_id, item)
        pl_count += 1
    print(f"    - Annual P&L: {pl_count} records")

    bs_count = 0
    for item in parsed_data.get('balance_sheet', []):
        _insert_balance_sheet(conn, org_id, item)
        bs_count += 1
    print(f"    - Balance Sheet: {bs_count} records")

    cie_count = 0
    for item in parsed_data.get('change_in_estimate', []):
        _insert_change_in_estimate(conn, org_id, item)
        cie_count += 1
    print(f"    - Change in Estimates: {cie_count} records")

    # Create chunks and embeddings
    print("  Creating chunks and embeddings...")
    chunks = chunk_pages(doc.texts, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    insert_chunks(
        conn,
        document_id=doc_id,
        chunks=[(c.index, c.text, c.page_start, c.page_end, c.token_count) for c in chunks],
    )
    print(f"    - Chunks: {len(chunks)}")

    # Embed and store in Qdrant
    qdrant = get_qdrant(s.qdrant_url, s.qdrant_api_key)
    batch = 64
    for i in range(0, len(chunks), batch):
        slice_chunks = chunks[i : i + batch]
        texts = [c.text for c in slice_chunks]
        vecs = embed_texts(texts, api_key=s.openai_api_key, model=s.embedding_model)

        if i == 0:
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

    print(f"    - Vectors: {len(chunks)} uploaded to Qdrant")

    conn.close()
    print("\n✓ Ingestion complete!")


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
