"""Vision-based ingestion agent using PyMuPDF + GPT-5 image analysis.

This agent scans PDFs under the configured data directory, converts each page to an image,
sends the images to GPT-5 for visual analysis, and extracts the following tables:

- organization
- profit_loss
- profit_loss_quarterly
- balance_sheet
- change_in_estimate
- ratios
- cash_flow
- shareholding_percentage
- price_performance

The vision-based approach leverages GPT-5's multimodal capabilities to directly analyze
PDF pages as images, which can be more accurate for complex table layouts than text extraction.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from openai import OpenAI

from .config import load_settings
from .db import (
    connect,
    ensure_schema,
    upsert_document,
    upsert_company,
    upsert_organization,
)

# Reuse helper insert functions for core sections
from .ingest import (
    _insert_profit_loss_quarterly,
    _insert_profit_loss,
    _insert_balance_sheet,
    _insert_change_in_estimate,
    _insert_ratios,
    _insert_cash_flow,
)

from .pdf_evaluator import evaluate_parser_output, print_evaluation_report
from .pdf_parser import extract_pdf
from .chunking import chunk_pages
from .embeddings import embed_texts
from .qdrant_store import get_qdrant, ensure_collection, upsert_points


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _iter_pdfs(root: Path) -> list[Path]:
    pdfs: list[Path] = []
    for p in sorted(root.glob("**/*.pdf")):
        pdfs.append(p)
    return pdfs


def _pdf_to_images(pdf_path: Path, dpi: int = 150) -> list[bytes]:
    """Convert PDF to a list of PNG images (one per page) using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for rendering (default 150 for good quality/speed balance)

    Returns:
        List of PNG image bytes, one per page
    """
    images: list[bytes] = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render page to pixmap at specified DPI
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 DPI is default
        pix = page.get_pixmap(matrix=mat)
        # Convert to PNG bytes
        png_bytes = pix.tobytes("png")
        images.append(png_bytes)

    doc.close()
    return images


def _encode_image(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _parse_with_gpt5_vision(images: list[bytes], company_hint: str | None = None) -> dict[str, Any]:
    """Use GPT-5 vision to parse financial data from PDF images.

    Sends all page images to GPT-5 and requests structured JSON extraction.
    """
    s = load_settings()
    client = OpenAI(api_key=s.openai_api_key)

    system_prompt = f"""You are a financial data extraction expert. Analyze the provided PDF page images and extract structured financial data.

Company hint: {company_hint or 'Unknown'}

Return a STRICT JSON object with the following shape (use null for missing values; do not add commentary):

{{
  "company_name": "string",
  "sector": "string|null",
  "industry": "string|null",

  "profit_loss_quarterly": [{{
    "period": "Q4FY24", "quarter": 4, "fiscal_year": 2024,
    "sales": number, "revenue": number, "ebitda": number, "ebitda_margin_pct": number,
    "depreciation": number, "ebit": number, "interest": number, "other_income": number,
    "exceptional_items": number, "pbt": number, "tax": number,
    "reported_pat": number, "adj_pat": number,
    "eps": number, "adj_eps": number, "shares_outstanding": number,
    "revenue_yoy_growth_pct": number, "revenue_qoq_growth_pct": number,
    "ebitda_yoy_growth_pct": number, "ebitda_qoq_growth_pct": number
  }}],

  "profit_loss": [{{
    "fiscal_year": "FY24",
    "revenue": number, "sales": number, "ebitda": number, "ebitda_margin_pct": number,
    "depreciation": number, "ebit": number, "interest": number, "other_income": number,
    "pbt": number, "tax": number, "tax_rate": number,
    "reported_pat": number, "adj_pat": number,
    "eps": number, "adj_eps": number, "shares_outstanding": number,
    "revenue_growth_pct": number, "ebitda_growth_pct": number, "pat_growth_pct": number
  }}],

  "balance_sheet": [{{
    "fiscal_year": "FY24",
    "cash": number, "accounts_receivable": number, "inventories": number, "other_current_assets": number,
    "investments": number, "gross_fixed_assets": number, "net_fixed_assets": number, "cwip": number,
    "intangible_assets": number, "total_assets": number,
    "accounts_payable": number, "short_term_debt": number, "long_term_debt": number, "total_liabilities": number,
    "share_capital": number, "reserves": number, "total_equity": number,
    "working_capital": number, "net_debt": number
  }}],

  "change_in_estimate": [{{
    "fiscal_year": "FY25E", "old_revenue": number, "new_revenue": number,
    "old_ebitda": number, "new_ebitda": number,
    "old_ebitda_margin_pct": number, "new_ebitda_margin_pct": number,
    "old_adj_pat": number, "new_adj_pat": number,
    "old_eps": number, "new_eps": number,
    "revenue_change_pct": number, "ebitda_change_pct": number, "pat_change_pct": number, "eps_change_pct": number
  }}],

  "ratios": [{{
    "fiscal_year": "FY24",
    "roe": number, "roa": number, "roce": number,
    "current_ratio": number, "quick_ratio": number,
    "debt_to_equity": number, "interest_coverage": number,
    "pe_ratio": number, "pb_ratio": number
  }}],

  "cash_flow": [{{
    "fiscal_year": "FY24",
    "operating_cash_flow": number, "investing_cash_flow": number, "capex": number,
    "financing_cash_flow": number, "free_cash_flow": number
  }}],

  "shareholding_percentage": [{{
    "as_of_date": "YYYY-MM-DD",
    "promoter_holding_pct": number, "public_holding_pct": number, "institutional_holding_pct": number,
    "fii_holding_pct": number, "dii_holding_pct": number, "retail_holding_pct": number,
    "pledged_shares_pct": number
  }}],

  "price_performance": [{{
    "date": "YYYY-MM-DD",
    "open_price": number, "high_price": number, "low_price": number, "close_price": number,
    "volume": number,
    "return_1d_pct": number, "return_1w_pct": number, "return_1m_pct": number,
    "return_3m_pct": number, "return_6m_pct": number, "return_1y_pct": number,
    "return_3y_pct": number, "return_5y_pct": number,
    "high_52w": number, "low_52w": number
  }}]
}}

INSTRUCTIONS:
1) Extract ALL available data from the images; use null if missing.
2) Convert percentage strings to numbers in percent units (e.g., 12.4% -> 12.4).
3) Remove thousands separators from numbers (e.g., 2,678 -> 2678).
4) Parse multiple fiscal years and quarters where present.
5) If a section is not present, return an empty array for that section.
6) Return ONLY valid JSON. No markdown fences or commentary.
7) Analyze ALL pages carefully - tables may span multiple pages.
"""

    # Build message content with all images
    content = [{"type": "text", "text": "Extract financial data from these PDF pages:"}]

    for idx, img_bytes in enumerate(images):
        base64_image = _encode_image(img_bytes)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"  # Use high detail for better table recognition
            }
        })

    content.append({
        "type": "text",
        "text": "Now extract the structured financial data as JSON following the schema above."
    })

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            max_completion_tokens=16000,  # Allow large responses for comprehensive extraction
        )

        text = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if text.startswith('```'):
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse GPT-5 response as JSON: {e}\nResponse: {text[:1000]}")

        return data

    except Exception as e:
        raise RuntimeError(f"GPT-5 vision API call failed: {e}")


# --------------------------------------------------------------------------------------
# Insert helpers for additional tables
# --------------------------------------------------------------------------------------

def _insert_shareholding_percentage(conn, org_id: str, item: dict) -> None:
    fields = {k: v for k, v in item.items() if k != 'as_of_date' and v is not None}
    if not fields:
        return
    columns = ', '.join(['organization_id', 'as_of_date'] + list(fields.keys()))
    placeholders = ', '.join(['%s'] * (2 + len(fields)))
    sql = f"INSERT INTO shareholding_percentage({columns}) VALUES ({placeholders})"
    with conn.cursor() as cur:
        cur.execute(sql, (org_id, item.get('as_of_date'), *fields.values()))
    conn.commit()


def _insert_price_performance(conn, org_id: str, item: dict) -> None:
    fields = {k: v for k, v in item.items() if k != 'date' and v is not None}
    if not fields:
        return
    columns = ', '.join(['organization_id', 'date'] + list(fields.keys()))
    placeholders = ', '.join(['%s'] * (2 + len(fields)))
    sql = f"INSERT INTO price_performance({columns}) VALUES ({placeholders})"
    with conn.cursor() as cur:
        cur.execute(sql, (org_id, item.get('date'), *fields.values()))
    conn.commit()


# --------------------------------------------------------------------------------------
# Public ingestion entry point
# --------------------------------------------------------------------------------------

def ingest_vision(max_files: int | None = None, *, file: str | None = None, no_db: bool = False, dpi: int = 150) -> None:
    """Ingest PDFs using GPT-5 vision-based extraction.

    Args:
        max_files: Limit number of PDFs to process
        file: Path to a single PDF to process (overrides max_files)
        no_db: If True, only parse and write JSON without database insertion
        dpi: Image resolution for PDF rendering (default 150)
    """
    s = load_settings()

    if file:
        target = Path(file)
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        pdfs = [target]
    else:
        pdfs = _iter_pdfs(s.data_dir)
        if max_files is not None:
            pdfs = pdfs[:max_files]

    if not pdfs:
        print(f"No PDFs found under {s.data_dir}")
        return

    conn = connect(s.database_url)
    ensure_schema(conn)

    # Qdrant client for vector storage
    qdrant = get_qdrant(s.qdrant_url, s.qdrant_api_key)
    main_collection = s.qdrant_collection
    company_collection = f"{s.qdrant_collection}-companies"
    main_vector_size: int | None = None
    company_vector_size: int | None = None

    success = 0
    errors = 0

    for idx, path in enumerate(pdfs, 1):
        try:
            print(f"[{idx}/{len(pdfs)}] Processing {path.name}")
            sha = _sha256_file(path)

            # Check if document already exists (deduplication)
            if not no_db:
                with conn.cursor() as cur:
                    cur.execute("SELECT id, sha256 FROM documents WHERE path = %s", (str(path),))
                    existing = cur.fetchone()
                    if existing:
                        existing_id, existing_sha = existing
                        if existing_sha == sha:
                            print(f"  ⊘ Skipped (already ingested with same content)")
                            success += 1
                            continue
                        else:
                            print(f"  ⚠ Document exists but SHA changed, re-ingesting...")

            # Create/update document record early (unless no_db)
            doc_id = None
            if not no_db:
                doc_id = upsert_document(conn, str(path), sha, None, None)

            # Convert PDF to images
            print(f"  → Converting PDF to images (DPI={dpi})...")
            images = _pdf_to_images(path, dpi=dpi)
            print(f"  → Converted {len(images)} pages")

            # Extract company hint from filename
            company_hint = path.stem.split('_')[0] if '_' in path.stem else path.stem

            # Parse with GPT-5 vision
            print(f"  → Analyzing with GPT-5 vision...")
            parsed = _parse_with_gpt5_vision(images, company_hint=company_hint)
            parsed['parsing_method'] = 'gpt5-vision'
            parsed['pdf_path'] = str(path)

            company_name = parsed.get('company_name') or company_hint
            sector = parsed.get('sector')
            industry = parsed.get('industry')

            org_id = None
            company_id = None
            if not no_db:
                company_id = upsert_company(conn, name=company_name, sector=sector, industry=industry, metadata={})
                org_id = upsert_organization(
                    conn,
                    name=company_name,
                    company_id=company_id,
                    document_id=doc_id,
                    sector=sector,
                    industry=industry,
                    pdf_path=str(path),
                    parsing_method='gpt5-vision',
                    metadata={},
                )

            # Insert sections
            if not no_db and org_id:
                for item in parsed.get('profit_loss_quarterly', []) or []:
                    _insert_profit_loss_quarterly(conn, org_id, item)
                for item in parsed.get('profit_loss', []) or []:
                    _insert_profit_loss(conn, org_id, item)
                for item in parsed.get('balance_sheet', []) or []:
                    _insert_balance_sheet(conn, org_id, item)
                for item in parsed.get('change_in_estimate', []) or []:
                    _insert_change_in_estimate(conn, org_id, item)
                for item in parsed.get('ratios', []) or []:
                    _insert_ratios(conn, org_id, item)
                for item in parsed.get('cash_flow', []) or []:
                    _insert_cash_flow(conn, org_id, item)
                for item in parsed.get('shareholding_percentage', []) or []:
                    _insert_shareholding_percentage(conn, org_id, item)
                for item in parsed.get('price_performance', []) or []:
                    _insert_price_performance(conn, org_id, item)

            # Build embeddings and upsert to Qdrant
            try:
                doc = extract_pdf(path)
                chunks = chunk_pages(doc.texts, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)

                batch = 64
                for i in range(0, len(chunks), batch):
                    slice_chunks = chunks[i : i + batch]
                    texts = [c.text for c in slice_chunks]
                    vecs = embed_texts(texts, api_key=s.openai_api_key, model=s.embedding_model)

                    if main_vector_size is None and vecs.shape[0] > 0:
                        main_vector_size = vecs.shape[1]
                        ensure_collection(qdrant, main_collection, main_vector_size)

                    ids = [f"{doc_id or 'nodoc'}:{c.index}" for c in slice_chunks]
                    payloads = [
                        {
                            "type": "chunk",
                            "document_id": doc_id,
                            "path": str(path),
                            "chunk_index": c.index,
                            "page_start": c.page_start,
                            "page_end": c.page_end,
                            "title": doc.title,
                            "text": c.text,
                            "company_name": company_name,
                        }
                        for c in slice_chunks
                    ]
                    upsert_points(qdrant, main_collection, ids, vecs.tolist(), payloads)
            except Exception as e:
                print(f"  ! Skipped Qdrant chunk embeddings: {e}")

            # Upsert company-name vector for fuzzy lookup
            try:
                if company_name:
                    vecs = embed_texts([company_name], api_key=s.openai_api_key, model=s.embedding_model)
                    if company_vector_size is None:
                        company_vector_size = vecs.shape[1]
                        ensure_collection(qdrant, company_collection, company_vector_size)
                    comp_id = f"company:{company_name}"
                    payload = {
                        "type": "company",
                        "name": company_name,
                        "company_id": company_id if not no_db else None,
                        "organization_id": org_id if not no_db else None,
                        "sector": sector,
                        "industry": industry,
                    }
                    upsert_points(qdrant, company_collection, [comp_id], vecs.tolist(), [payload])
            except Exception as e:
                print(f"  ! Skipped Qdrant company embedding: {e}")

            # Dump parsed JSON for inspection
            try:
                out_json = Path.cwd() / f"parsed_output_vision_{path.stem}.json"
                with open(out_json, 'w') as f:
                    json.dump(parsed, f, indent=2, default=str)
            except Exception:
                pass

            if no_db:
                print(f"  ✓ Parsed {company_name} (no DB)")
            else:
                print(f"  ✓ Ingested {company_name}")
            success += 1

        except Exception as e:
            print(f"  ✗ Failed {path.name}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1

    print(f"Done. Success={success}, Errors={errors}")


def eval_vision(pdf_path: str, csv_paths: list[str], dpi: int = 150) -> dict[str, Any]:
    """Run vision-based extraction for a single PDF and evaluate against CSVs."""
    p = Path(pdf_path)
    company_hint = p.stem

    # Convert and parse
    images = _pdf_to_images(p, dpi=dpi)
    parsed = _parse_with_gpt5_vision(images, company_hint=company_hint)

    # Save for inspection
    out_json = Path.cwd() / f"parsed_output_vision_{p.stem}.json"
    with open(out_json, 'w') as f:
        json.dump(parsed, f, indent=2, default=str)

    results = evaluate_parser_output(parsed, csv_paths)
    print_evaluation_report(results)

    # Save evaluation results
    eval_out = Path.cwd() / f"eval_results_vision_{p.stem}.json"
    with open(eval_out, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="ingest_vision", description="Ingest PDFs using GPT-5 vision")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of PDFs to process")
    parser.add_argument("--file", type=str, default=None, help="Path to a single PDF to process")
    parser.add_argument("--no-db", action="store_true", help="Do not write to DB; only parse and write JSON")
    parser.add_argument("--dpi", type=int, default=150, help="Image resolution for PDF rendering (default 150)")
    parser.add_argument("--eval", action="store_true", help="Run evaluation against CSV ground truth")
    parser.add_argument("--eval-csv", action="append", default=None, help="Path to a CSV file (repeatable)")
    args = parser.parse_args()

    if args.eval:
        if not args.file:
            raise SystemExit("--eval requires --file to be specified")
        if not args.eval_csv:
            raise SystemExit("--eval requires at least one --eval-csv path")
        eval_vision(args.file, args.eval_csv, dpi=args.dpi)
    else:
        ingest_vision(max_files=args.max_files, file=args.file, no_db=args.no_db, dpi=args.dpi)
