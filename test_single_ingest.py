#!/usr/bin/env python3
"""Simple test to ingest a single PDF into the database."""

import sys
from pathlib import Path

from geojit.config import load_settings
from geojit.db import connect, upsert_company, upsert_organization
from geojit.pdf_parser_text import parse_pdf_text_based

# Simple helper functions to insert data
def insert_profit_loss_quarterly(conn, org_id: str, data: dict) -> None:
    """Insert quarterly P&L data."""
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
    """Insert annual P&L data."""
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
    """Insert balance sheet data."""
    fields = {k: v for k, v in data.items() if k != 'fiscal_year' and v is not None}
    if not fields:
        return

    columns = ['organization_id', 'fiscal_year'] + list(fields.keys())
    placeholders = ['%s'] * len(columns)
    sql = f"INSERT INTO balance_sheet({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

    with conn.cursor() as cur:
        cur.execute(sql, [org_id, data.get('fiscal_year')] + list(fields.values()))
    conn.commit()


def main():
    # Use the TTK PDF we already tested
    pdf_path = "/Users/cameronellis/work/geojit-research-cli/Financial_Research_Agent_Files/SP20241406115209223TTK.pdf"

    if not Path(pdf_path).exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    print("="*80)
    print("SINGLE PDF INGESTION TEST")
    print("="*80)
    print(f"\nPDF: {Path(pdf_path).name}\n")

    # Step 1: Parse PDF
    print("[1/4] Parsing PDF with text-based parser...")
    try:
        parsed_data = parse_pdf_text_based(pdf_path, company_hint="TTK")
        company_name = parsed_data.get('company_name', 'TTK')
        print(f"  ✓ Parsed data for: {company_name}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Connect to database
    print("\n[2/4] Connecting to database...")
    try:
        s = load_settings()
        conn = connect(s.database_url)
        print(f"  ✓ Connected to: {s.database_url}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        sys.exit(1)

    # Step 3: Create company and organization
    print("\n[3/4] Creating company and organization records...")
    try:
        company_id = upsert_company(
            conn,
            name=company_name,
            sector=parsed_data.get('sector'),
            industry=parsed_data.get('industry'),
            metadata={}
        )
        print(f"  ✓ Company ID: {company_id}")

        org_id = upsert_organization(
            conn,
            name=company_name,
            company_id=company_id,
            document_id=None,  # Skip document for now
            sector=parsed_data.get('sector'),
            industry=parsed_data.get('industry'),
            pdf_path=pdf_path,
            parsing_method='text',
            metadata={}
        )
        print(f"  ✓ Organization ID: {org_id}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Insert financial data
    print("\n[4/4] Inserting financial data...")
    try:
        # Quarterly P&L
        quarterly = parsed_data.get('profit_loss_quarterly', [])
        for item in quarterly:
            insert_profit_loss_quarterly(conn, org_id, item)
        print(f"  ✓ Inserted {len(quarterly)} quarterly P&L records")

        # Annual P&L
        annual = parsed_data.get('profit_loss', [])
        for item in annual:
            insert_profit_loss(conn, org_id, item)
        print(f"  ✓ Inserted {len(annual)} annual P&L records")

        # Balance Sheet
        balance = parsed_data.get('balance_sheet', [])
        for item in balance:
            insert_balance_sheet(conn, org_id, item)
        print(f"  ✓ Inserted {len(balance)} balance sheet records")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Verify data
    print("\n[VERIFICATION] Checking inserted data...")
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM organization")
        org_count = cur.fetchone()[0]
        print(f"  Organizations: {org_count}")

        cur.execute("SELECT COUNT(*) FROM profit_loss_quarterly")
        plq_count = cur.fetchone()[0]
        print(f"  Quarterly P&L: {plq_count}")

        cur.execute("SELECT COUNT(*) FROM profit_loss")
        pl_count = cur.fetchone()[0]
        print(f"  Annual P&L: {pl_count}")

        cur.execute("SELECT COUNT(*) FROM balance_sheet")
        bs_count = cur.fetchone()[0]
        print(f"  Balance Sheet: {bs_count}")

    print("\n" + "="*80)
    print("✅ INGESTION SUCCESSFUL")
    print("="*80)


if __name__ == "__main__":
    main()
