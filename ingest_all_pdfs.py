#!/usr/bin/env python3
"""Batch ingest all PDFs with logging."""

import sys
from pathlib import Path
from tqdm import tqdm

from geojit.config import load_settings
from geojit.db import connect, upsert_company, upsert_organization
from geojit.pdf_parser_text import parse_pdf_text_based


def insert_profit_loss_quarterly(conn, org_id: str, data: dict) -> None:
    fields = {k: v for k, v in data.items() if k not in ['period', 'quarter', 'fiscal_year'] and v is not None}
    if not fields:
        return
    columns = ['organization_id', 'period', 'quarter', 'fiscal_year'] + list(fields.keys())
    placeholders = ['%s'] * len(columns)
    sql = f"INSERT INTO profit_loss_quarterly({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
    with conn.cursor() as cur:
        cur.execute(sql, [org_id, data.get('period'), data.get('quarter'), data.get('fiscal_year')] + list(fields.values()))
    conn.commit()


def insert_profit_loss(conn, org_id: str, data: dict) -> None:
    fields = {k: v for k, v in data.items() if k != 'fiscal_year' and v is not None}
    if not fields:
        return
    columns = ['organization_id', 'fiscal_year'] + list(fields.keys())
    placeholders = ['%s'] * len(columns)
    sql = f"INSERT INTO profit_loss({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
    with conn.cursor() as cur:
        cur.execute(sql, [org_id, data.get('fiscal_year')] + list(fields.values()))
    conn.commit()


def insert_balance_sheet(conn, org_id: str, data: dict) -> None:
    fields = {k: v for k, v in data.items() if k != 'fiscal_year' and v is not None}
    if not fields:
        return
    columns = ['organization_id', 'fiscal_year'] + list(fields.keys())
    placeholders = ['%s'] * len(columns)
    sql = f"INSERT INTO balance_sheet({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
    with conn.cursor() as cur:
        cur.execute(sql, [org_id, data.get('fiscal_year')] + list(fields.values()))
    conn.commit()


def insert_change_in_estimate(conn, org_id: str, data: dict) -> None:
    fields = {k: v for k, v in data.items() if k != 'fiscal_year' and v is not None}
    if not fields:
        return
    columns = ['organization_id', 'fiscal_year'] + list(fields.keys())
    placeholders = ['%s'] * len(columns)
    sql = f"INSERT INTO change_in_estimate({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
    with conn.cursor() as cur:
        cur.execute(sql, [org_id, data.get('fiscal_year')] + list(fields.values()))
    conn.commit()


def main():
    s = load_settings()
    data_dir = Path(s.data_dir)

    # Find all PDFs
    pdfs = sorted(data_dir.glob("**/*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {data_dir}\n")

    if not pdfs:
        print("No PDFs found!")
        sys.exit(1)

    # Connect to database
    conn = connect(s.database_url)
    print(f"Connected to database: {s.database_url}\n")

    success_count = 0
    error_count = 0
    errors = []

    for pdf_path in tqdm(pdfs, desc="Ingesting PDFs"):
        try:
            # Parse PDF
            company_hint = pdf_path.stem.split('_')[0] if '_' in pdf_path.stem else pdf_path.stem
            parsed_data = parse_pdf_text_based(str(pdf_path), company_hint=company_hint)

            company_name = parsed_data.get('company_name', company_hint)

            # Create company
            company_id = upsert_company(
                conn,
                name=company_name,
                sector=parsed_data.get('sector'),
                industry=parsed_data.get('industry'),
                metadata={}
            )

            # Create organization
            org_id = upsert_organization(
                conn,
                name=company_name,
                company_id=company_id,
                document_id=None,
                sector=parsed_data.get('sector'),
                industry=parsed_data.get('industry'),
                pdf_path=str(pdf_path),
                parsing_method='text',
                metadata={}
            )

            # Insert data
            for item in parsed_data.get('profit_loss_quarterly', []):
                insert_profit_loss_quarterly(conn, org_id, item)

            for item in parsed_data.get('profit_loss', []):
                insert_profit_loss(conn, org_id, item)

            for item in parsed_data.get('balance_sheet', []):
                insert_balance_sheet(conn, org_id, item)

            for item in parsed_data.get('change_in_estimate', []):
                insert_change_in_estimate(conn, org_id, item)

            success_count += 1

        except Exception as e:
            error_count += 1
            errors.append((pdf_path.name, str(e)))

    # Summary
    print("\n" + "="*80)
    print(f"INGESTION COMPLETE")
    print(f"  Success: {success_count}/{len(pdfs)}")
    print(f"  Errors:  {error_count}/{len(pdfs)}")

    if errors:
        print(f"\nErrors:")
        for filename, error in errors[:10]:  # Show first 10 errors
            print(f"  - {filename}: {error}")

    print("="*80)


if __name__ == "__main__":
    main()
