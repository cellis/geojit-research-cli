#!/usr/bin/env python3
"""
Comparison script for ingestion validation.
Runs ingestion against geojit-test database and validates results against eval CSVs.
"""
import sys
import csv
import psycopg
from pathlib import Path
from typing import Any
from decimal import Decimal

DATABASE_URL = "postgresql://localhost/geojit-test"

# Eval CSV files
EVAL_FILES = {
    "quarterly_profit_loss": "/Users/cameronellis/work/geojit-research-cli/SP20241406115209223TTK - quarterly_profit_loss.csv",
    "balance_sheet": "/Users/cameronellis/work/geojit-research-cli/SP20241406115209223TTK - balance_sheet.csv",
    "change_in_estimates": "/Users/cameronellis/work/geojit-research-cli/SP20241406115209223TTK - change_in_estimates.csv",
    "profit_loss": "/Users/cameronellis/work/geojit-research-cli/SP20241406115209223TTK - profit_loss.csv",
}


def parse_value(value_str: str) -> float | None:
    """Parse numeric values from CSV, handling various formats."""
    if not value_str or value_str.strip() == '' or value_str.strip() == 'NA':
        return None

    # Remove commas and quotes
    clean = value_str.strip().replace(',', '').replace('"', '')

    # Handle percentages
    if '%' in clean or 'bps' in clean:
        # Don't parse percentages for now, just return None
        return None

    try:
        return float(clean)
    except ValueError:
        return None


def load_quarterly_profit_loss_eval(csv_path: str) -> dict[str, dict]:
    """Load quarterly P&L eval data."""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row['Rs (cr)'].strip()
            for period in ['Q4FY24', 'Q4FY23', 'Q3FY24', 'FY24', 'FY23']:
                if period not in data:
                    data[period] = {}
                value = parse_value(row.get(period, ''))
                if value is not None:
                    data[period][metric.lower().replace(' ', '_')] = value
    return data


def load_balance_sheet_eval(csv_path: str) -> dict[str, dict]:
    """Load balance sheet eval data."""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row['Y.E March (Rs. cr)'].strip()
            for fy in ['FY21A', 'FY22A', 'FY23A', 'FY24', 'FY25E', 'FY26E']:
                if fy not in data:
                    data[fy] = {}
                value = parse_value(row.get(fy, ''))
                if value is not None:
                    data[fy][metric.lower().replace(' ', '_').replace('.', '')] = value
    return data


def load_change_in_estimates_eval(csv_path: str) -> dict[str, dict]:
    """Load change in estimates eval data."""
    data = {'old': {}, 'new': {}, 'change': {}}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get(''):
                continue
            metric = row[''].strip()

            # Old estimates
            for fy in ['FY25E', 'FY26E']:
                if f'old_{fy}' not in data['old']:
                    data['old'][f'old_{fy}'] = {}
                value = parse_value(row.get('Old estimates', ''))
                if value is not None:
                    data['old'][f'old_{fy}'][metric.lower().replace(' ', '_')] = value

            # New estimates
            for fy in ['FY25E', 'FY26E']:
                if f'new_{fy}' not in data['new']:
                    data['new'][f'new_{fy}'] = {}
                value = parse_value(row.get('New estimates', ''))
                if value is not None:
                    data['new'][f'new_{fy}'][metric.lower().replace(' ', '_')] = value

    return data


def load_profit_loss_eval(csv_path: str) -> dict[str, dict]:
    """Load annual P&L eval data."""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row['Y.E March (Rs. cr)'].strip()
            for fy in ['FY21A', 'FYZZA', 'FY23A', 'FY24A', 'FY25E', 'FY26E']:
                # Normalize FYZZA to FY22A first
                normalized_fy = 'FY22A' if fy == 'FYZZA' else fy
                if normalized_fy not in data:
                    data[normalized_fy] = {}
                value = parse_value(row.get(fy, ''))
                if value is not None:
                    data[normalized_fy][metric.lower().replace(' ', '_').replace('%', 'pct')] = value
    return data


def compare_values(expected: float, actual: float | Decimal | None, tolerance: float = 0.01) -> tuple[bool, str]:
    """Compare two numeric values with tolerance."""
    if actual is None:
        return False, f"Missing (expected {expected})"

    actual_float = float(actual) if isinstance(actual, Decimal) else actual
    diff = abs(expected - actual_float)

    # Use relative tolerance for large numbers
    rel_tolerance = abs(expected * tolerance)
    if diff <= max(tolerance, rel_tolerance):
        return True, "✓"
    else:
        return False, f"✗ Expected {expected}, got {actual_float} (diff: {diff:.2f})"


def query_quarterly_profit_loss(conn: psycopg.Connection, org_id: str, period: str) -> dict:
    """Query quarterly P&L data from database."""
    sql = """
        SELECT sales, revenue, ebitda, depreciation, ebit, interest, pbt, tax, reported_pat, adj_pat, eps,
               ebitda_margin_pct, revenue_yoy_growth_pct, revenue_qoq_growth_pct
        FROM profit_loss_quarterly
        WHERE organization_id = %s AND period = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (org_id, period))
        row = cur.fetchone()
        if not row:
            return {}

        columns = [desc[0] for desc in cur.description]
        return {col: val for col, val in zip(columns, row) if val is not None}


def query_balance_sheet(conn: psycopg.Connection, org_id: str, fiscal_year: str) -> dict:
    """Query balance sheet data from database."""
    # Try both formats: FY24A and FY24
    fiscal_year_alt = fiscal_year.rstrip('AE')  # Remove trailing A or E
    sql = """
        SELECT cash, accounts_receivable, receivables, inventories, other_current_assets,
               investments, gross_fixed_assets, net_fixed_assets, cwip, intangible_assets,
               total_current_assets, total_assets, accounts_payable, payables,
               short_term_debt, long_term_debt, total_current_liabilities, total_liabilities,
               share_capital, reserves, total_equity, net_worth, working_capital, net_debt
        FROM balance_sheet
        WHERE organization_id = %s AND (fiscal_year = %s OR fiscal_year = %s)
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (org_id, fiscal_year, fiscal_year_alt))
        row = cur.fetchone()
        if not row:
            return {}

        columns = [desc[0] for desc in cur.description]
        return {col: val for col, val in zip(columns, row) if val is not None}


def query_profit_loss(conn: psycopg.Connection, org_id: str, fiscal_year: str) -> dict:
    """Query annual P&L data from database."""
    # Try both formats: FY24A and FY24
    fiscal_year_alt = fiscal_year.rstrip('AE')  # Remove trailing A or E
    sql = """
        SELECT revenue, sales, ebitda, depreciation, ebit, interest, other_income,
               pbt, tax, reported_pat, adj_pat, eps, shares_outstanding,
               revenue_growth_pct, ebitda_growth_pct, ebitda_margin_pct
        FROM profit_loss
        WHERE organization_id = %s AND (fiscal_year = %s OR fiscal_year = %s)
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (org_id, fiscal_year, fiscal_year_alt))
        row = cur.fetchone()
        if not row:
            return {}

        columns = [desc[0] for desc in cur.description]
        return {col: val for col, val in zip(columns, row) if val is not None}


def get_organization_id(conn: psycopg.Connection, name: str) -> str | None:
    """Get organization ID by name."""
    sql = "SELECT id FROM organization WHERE LOWER(name) = LOWER(%s) LIMIT 1"
    with conn.cursor() as cur:
        cur.execute(sql, (name,))
        row = cur.fetchone()
        return str(row[0]) if row else None


def run_comparison():
    """Run the comparison workflow."""
    print("=" * 80)
    print("Ingestion Comparison Tool")
    print("=" * 80)

    # Load eval data
    print("\n1. Loading evaluation data from CSVs...")
    quarterly_pl_eval = load_quarterly_profit_loss_eval(EVAL_FILES['quarterly_profit_loss'])
    balance_sheet_eval = load_balance_sheet_eval(EVAL_FILES['balance_sheet'])
    profit_loss_eval = load_profit_loss_eval(EVAL_FILES['profit_loss'])
    print(f"   ✓ Loaded quarterly P&L: {len(quarterly_pl_eval)} periods")
    print(f"   ✓ Loaded balance sheet: {len(balance_sheet_eval)} fiscal years")
    print(f"   ✓ Loaded annual P&L: {len(profit_loss_eval)} fiscal years")

    # Connect to test database
    print(f"\n2. Connecting to database: {DATABASE_URL}")
    conn = psycopg.connect(DATABASE_URL)

    # Assume company name - extract from filename
    # Try both variations
    company_name = "TTK Prestige Ltd."
    org_id = get_organization_id(conn, company_name)
    if not org_id:
        company_name = "TTK"
        org_id = get_organization_id(conn, company_name)

    if not org_id:
        print(f"\n   ✗ Organization '{company_name}' not found in database!")
        print(f"   Run ingestion first or check company name.")
        conn.close()
        return 1

    print(f"   ✓ Found organization: {company_name} (ID: {org_id})")

    # Compare quarterly P&L
    print("\n3. Comparing Quarterly Profit & Loss...")
    qpl_passed = 0
    qpl_failed = 0
    for period, expected_data in quarterly_pl_eval.items():
        if period in ['FY24', 'FY23']:  # Skip annual data in quarterly comparison
            continue
        actual_data = query_quarterly_profit_loss(conn, org_id, period)
        if not actual_data:
            print(f"   ✗ {period}: No data found in database")
            qpl_failed += 1
            continue

        period_passed = True
        for metric, expected_value in expected_data.items():
            # Map eval metric names to DB column names
            if metric == 'ebitda_margins':
                metric = 'ebitda_margin_pct'

            actual_value = actual_data.get(metric)
            passed, message = compare_values(expected_value, actual_value)
            if not passed:
                print(f"   {period}.{metric}: {message}")
                period_passed = False

        if period_passed:
            qpl_passed += 1
        else:
            qpl_failed += 1

    print(f"   Summary: {qpl_passed} passed, {qpl_failed} failed")

    # Compare annual P&L
    print("\n4. Comparing Annual Profit & Loss...")
    apl_passed = 0
    apl_failed = 0
    for fiscal_year, expected_data in profit_loss_eval.items():
        actual_data = query_profit_loss(conn, org_id, fiscal_year)
        if not actual_data:
            print(f"   ✗ {fiscal_year}: No data found in database")
            apl_failed += 1
            continue

        fy_passed = True
        for metric, expected_value in expected_data.items():
            # Skip change/growth metrics for now
            if 'change' in metric or 'pct' in metric:
                continue

            actual_value = actual_data.get(metric)
            passed, message = compare_values(expected_value, actual_value)
            if not passed:
                print(f"   {fiscal_year}.{metric}: {message}")
                fy_passed = False

        if fy_passed:
            apl_passed += 1
        else:
            apl_failed += 1

    print(f"   Summary: {apl_passed} passed, {apl_failed} failed")

    # Compare balance sheet
    print("\n5. Comparing Balance Sheet...")
    bs_passed = 0
    bs_failed = 0
    for fiscal_year, expected_data in balance_sheet_eval.items():
        actual_data = query_balance_sheet(conn, org_id, fiscal_year)
        if not actual_data:
            print(f"   ✗ {fiscal_year}: No data found in database")
            bs_failed += 1
            continue

        fy_passed = True
        for metric, expected_value in expected_data.items():
            actual_value = actual_data.get(metric)
            if actual_value is None:
                # Try alternate column names
                if metric == 'accounts_receivable':
                    actual_value = actual_data.get('receivables')
                elif metric == 'other_cur_assets':
                    actual_value = actual_data.get('other_current_assets')

            passed, message = compare_values(expected_value, actual_value)
            if not passed:
                print(f"   {fiscal_year}.{metric}: {message}")
                fy_passed = False

        if fy_passed:
            bs_passed += 1
        else:
            bs_failed += 1

    print(f"   Summary: {bs_passed} passed, {bs_failed} failed")

    # Final summary
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    total_passed = qpl_passed + apl_passed + bs_passed
    total_failed = qpl_failed + apl_failed + bs_failed
    total_tests = total_passed + total_failed

    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_passed} ({100*total_passed/total_tests if total_tests > 0 else 0:.1f}%)")
    print(f"Failed: {total_failed} ({100*total_failed/total_tests if total_tests > 0 else 0:.1f}%)")

    conn.close()

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_comparison())
