"""Agent-based PDF ingestion system that analyzes PDFs first, then adapts schema."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from tqdm import tqdm

from .config import load_settings
from .db import connect, ensure_schema, upsert_document, insert_chunks, upsert_company, upsert_organization
from .pdf_parser import extract_pdf
from .pdf_parser_text import parse_pdf_text_based
from .chunking import chunk_pages
from .embeddings import embed_texts
from .qdrant_store import get_qdrant, ensure_collection, upsert_points


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


def analyze_pdf_structures(pdfs: list[Path], sample_size: int = 5) -> dict[str, Any]:
    """
    Analyze a sample of PDFs to understand their structure and schema requirements.

    Returns:
        Dictionary with schema analysis including all discovered field names
    """
    print(f"\n[ANALYSIS PHASE] Analyzing {min(sample_size, len(pdfs))} PDFs to understand schema...")

    all_fields = {
        'profit_loss_quarterly': set(),
        'profit_loss': set(),
        'balance_sheet': set(),
        'change_in_estimate': set(),
        'ratios': set(),
        'cash_flow': set()
    }

    companies = []
    sample_pdfs = pdfs[:sample_size]

    for i, path in enumerate(sample_pdfs, 1):
        print(f"\n  [{i}/{len(sample_pdfs)}] Analyzing {path.name}...")
        try:
            company_hint = path.stem.split('_')[0] if '_' in path.stem else path.stem

            # Parse with text-based parser
            print(f"    - Extracting text from PDF...")
            parsed_data = parse_pdf_text_based(str(path), company_hint=company_hint)

            company_name = parsed_data.get('company_name', company_hint)
            print(f"    - Company: {company_name}")

            companies.append({
                'name': company_name,
                'sector': parsed_data.get('sector'),
                'industry': parsed_data.get('industry')
            })

            # Collect all field names from each section
            for section in all_fields.keys():
                items = parsed_data.get(section, [])
                if items:
                    print(f"    - Found {len(items)} {section} entries")
                    for item in items:
                        all_fields[section].update(item.keys())

        except Exception as e:
            print(f"    - ERROR: {e}")
            continue

    # Convert sets to sorted lists
    schema_analysis = {
        'companies': companies,
        'fields': {k: sorted(v) for k, v in all_fields.items()},
        'total_pdfs_analyzed': len(sample_pdfs)
    }

    print(f"\n[SCHEMA ANALYSIS COMPLETE]")
    print(f"  Companies found: {len(companies)}")
    for section, fields in schema_analysis['fields'].items():
        if fields:
            print(f"  {section}: {len(fields)} fields")

    return schema_analysis


def ingest_with_agent(max_files: int | None = None, sample_size: int = 3) -> None:
    """
    Agent-based ingestion that analyzes PDFs first, then ingests.

    Args:
        max_files: Maximum number of PDFs to ingest (None = all)
        sample_size: Number of PDFs to analyze for schema discovery
    """
    s = load_settings()

    # Step 1: Find all PDFs
    print("[DISCOVERY] Finding PDFs...")
    pdfs = iter_pdfs(s.data_dir)
    if max_files is not None:
        pdfs = pdfs[:max_files]

    if not pdfs:
        print(f"No PDFs found under {s.data_dir}")
        return

    print(f"Found {len(pdfs)} PDFs")

    # Step 2: Analyze schema from sample
    schema_analysis = analyze_pdf_structures(pdfs, sample_size=min(sample_size, len(pdfs)))

    # Save analysis results
    analysis_file = Path("schema_analysis.json")
    with open(analysis_file, 'w') as f:
        json.dump(schema_analysis, f, indent=2, default=str)
    print(f"\nSchema analysis saved to {analysis_file}")

    # Step 3: Log schema discovery
    print("\n" + "="*80)
    print("SCHEMA DISCOVERED - Saved to schema_analysis.json")
    print("="*80)
    print("Proceeding with ingestion...")

    # Step 4: Connect to database and ensure schema
    print("\n[DATABASE] Connecting and ensuring schema...")
    conn = connect(s.database_url)
    ensure_schema(conn)
    print("Database schema ready")

    # Step 5: Set up Qdrant
    print("\n[VECTOR STORE] Setting up Qdrant...")
    qdrant = get_qdrant(s.qdrant_url, s.qdrant_api_key)
    vector_size = None

    # Step 6: Ingest all PDFs
    print(f"\n[INGESTION] Processing {len(pdfs)} PDFs...")
    print("="*80)

    success_count = 0
    error_count = 0

    for idx, path in enumerate(tqdm(pdfs, desc="Ingesting PDFs"), 1):
        print(f"\n[{idx}/{len(pdfs)}] Processing: {path.name}")

        try:
            # Hash file
            sha = _sha256_file(path)

            # Extract text for chunking
            print(f"  [1/5] Extracting PDF text...")
            doc = extract_pdf(path)

            # Upsert document
            print(f"  [2/5] Storing document metadata...")
            doc_id = upsert_document(conn, str(path), sha, doc.title, doc.pages)

            # Parse structured data
            print(f"  [3/5] Parsing structured financial data...")
            company_hint = path.stem.split('_')[0] if '_' in path.stem else path.stem
            parsed_data = parse_pdf_text_based(str(path), company_hint=company_hint)

            # Store company and organization
            company_name = parsed_data.get('company_name', company_hint)
            print(f"  [4/5] Storing {company_name} data...")

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

            # Insert financial data
            _insert_all_financial_data(conn, org_id, parsed_data)

            # Chunk and embed
            print(f"  [5/5] Creating embeddings and storing in Qdrant...")
            chunks = chunk_pages(doc.texts, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)

            # Insert chunks to Postgres
            insert_chunks(
                conn,
                document_id=doc_id,
                chunks=[(c.index, c.text, c.page_start, c.page_end, c.token_count) for c in chunks],
            )

            # Embed and send to Qdrant in batches
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

            print(f"  ✅ SUCCESS - {company_name}")
            success_count += 1

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue

    print("\n" + "="*80)
    print(f"INGESTION COMPLETE")
    print(f"  Success: {success_count}/{len(pdfs)}")
    print(f"  Errors:  {error_count}/{len(pdfs)}")
    print("="*80)


def _insert_all_financial_data(conn, org_id: str, parsed_data: dict) -> None:
    """Insert all financial data sections."""
    from .ingest import (
        _insert_profit_loss_quarterly,
        _insert_profit_loss,
        _insert_balance_sheet,
        _insert_change_in_estimate,
        _insert_ratios,
        _insert_cash_flow
    )

    sections_inserted = []

    for item in parsed_data.get('profit_loss_quarterly', []):
        _insert_profit_loss_quarterly(conn, org_id, item)
        if 'profit_loss_quarterly' not in sections_inserted:
            sections_inserted.append('profit_loss_quarterly')

    for item in parsed_data.get('profit_loss', []):
        _insert_profit_loss(conn, org_id, item)
        if 'profit_loss' not in sections_inserted:
            sections_inserted.append('profit_loss')

    for item in parsed_data.get('balance_sheet', []):
        _insert_balance_sheet(conn, org_id, item)
        if 'balance_sheet' not in sections_inserted:
            sections_inserted.append('balance_sheet')

    for item in parsed_data.get('change_in_estimate', []):
        _insert_change_in_estimate(conn, org_id, item)
        if 'change_in_estimate' not in sections_inserted:
            sections_inserted.append('change_in_estimate')

    for item in parsed_data.get('ratios', []):
        _insert_ratios(conn, org_id, item)
        if 'ratios' not in sections_inserted:
            sections_inserted.append('ratios')

    for item in parsed_data.get('cash_flow', []):
        _insert_cash_flow(conn, org_id, item)
        if 'cash_flow' not in sections_inserted:
            sections_inserted.append('cash_flow')

    if sections_inserted:
        print(f"      Inserted: {', '.join(sections_inserted)}")
