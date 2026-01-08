from __future__ import annotations

import psycopg
from dataclasses import dataclass


@dataclass
class DocRow:
    id: int
    path: str
    sha256: str
    title: str | None
    pages: int | None


def connect(database_url: str) -> psycopg.Connection:
    return psycopg.connect(database_url)


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    sha256 TEXT NOT NULL,
    title TEXT,
    pages INT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    text TEXT NOT NULL,
    page_start INT,
    page_end INT,
    token_count INT,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    sector TEXT,
    industry TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_companies_sector ON companies(sector);
CREATE INDEX IF NOT EXISTS idx_companies_name_lower ON companies(LOWER(name));

CREATE TABLE IF NOT EXISTS financial_metrics (
    id BIGSERIAL PRIMARY KEY,
    company_id INT REFERENCES companies(id) ON DELETE CASCADE,
    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    metric_value NUMERIC,
    metric_value_text TEXT,
    unit TEXT,
    period TEXT,
    quarter INT,
    fiscal_year INT,
    report_date DATE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_metrics_company ON financial_metrics(company_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON financial_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_period ON financial_metrics(period);
CREATE INDEX IF NOT EXISTS idx_metrics_fiscal_year ON financial_metrics(fiscal_year);
CREATE INDEX IF NOT EXISTS idx_metrics_company_metric ON financial_metrics(company_id, metric_name);

-- New organization-centric tables
CREATE TABLE IF NOT EXISTS organization (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    company_id INT REFERENCES companies(id) ON DELETE CASCADE,
    document_id INT REFERENCES documents(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    sector TEXT,
    industry TEXT,
    pdf_path TEXT,
    parsing_method TEXT, -- 'image' or 'text'
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_organization_company ON organization(company_id);
CREATE INDEX IF NOT EXISTS idx_organization_name ON organization(LOWER(name));

-- Profit & Loss (Annual)
CREATE TABLE IF NOT EXISTS profit_loss (
    id BIGSERIAL PRIMARY KEY,
    organization_id UUID REFERENCES organization(id) ON DELETE CASCADE,
    fiscal_year TEXT NOT NULL,

    -- Revenue metrics (multiple naming conventions)
    revenue NUMERIC,
    sales NUMERIC,
    total_income NUMERIC,
    turnover NUMERIC,

    -- EBITDA metrics
    ebitda NUMERIC,
    operating_profit_before_depreciation NUMERIC,

    -- Depreciation
    depreciation NUMERIC,
    amortization NUMERIC,

    -- EBIT
    ebit NUMERIC,
    operating_profit NUMERIC,

    -- Interest
    interest NUMERIC,
    interest_expense NUMERIC,
    finance_cost NUMERIC,

    -- Other income
    other_income NUMERIC,
    non_operating_income NUMERIC,

    -- Exceptional items
    exceptional_items NUMERIC,
    extraordinary_items NUMERIC,

    -- PBT
    pbt NUMERIC,
    profit_before_tax NUMERIC,

    -- Tax
    tax NUMERIC,
    tax_expense NUMERIC,
    income_tax NUMERIC,
    tax_rate NUMERIC,

    -- Share of profit from associates
    share_of_profit_associates NUMERIC,

    -- Minority interest
    minority_interest NUMERIC,

    -- PAT
    reported_pat NUMERIC,
    net_profit NUMERIC,
    profit_after_tax NUMERIC,

    -- Adjusted PAT
    adj_pat NUMERIC,
    adjusted_pat NUMERIC,
    normalized_pat NUMERIC,

    -- Adjustments
    adjustments NUMERIC,

    -- EPS
    eps NUMERIC,
    adj_eps NUMERIC,
    basic_eps NUMERIC,
    diluted_eps NUMERIC,

    -- Shares
    shares_outstanding NUMERIC,
    no_of_shares NUMERIC,

    -- Growth percentages
    revenue_growth_pct NUMERIC,
    ebitda_growth_pct NUMERIC,
    pat_growth_pct NUMERIC,

    -- Margins
    ebitda_margin_pct NUMERIC,
    operating_margin_pct NUMERIC,
    net_margin_pct NUMERIC,

    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_profit_loss_org ON profit_loss(organization_id);
CREATE INDEX IF NOT EXISTS idx_profit_loss_fiscal_year ON profit_loss(fiscal_year);

-- Profit & Loss Quarterly
CREATE TABLE IF NOT EXISTS profit_loss_quarterly (
    id BIGSERIAL PRIMARY KEY,
    organization_id UUID REFERENCES organization(id) ON DELETE CASCADE,
    period TEXT NOT NULL, -- e.g., 'Q4FY24'
    quarter INT,
    fiscal_year INT,

    -- Same columns as profit_loss for consistency
    revenue NUMERIC,
    sales NUMERIC,
    total_income NUMERIC,
    turnover NUMERIC,

    ebitda NUMERIC,
    operating_profit_before_depreciation NUMERIC,

    depreciation NUMERIC,
    amortization NUMERIC,

    ebit NUMERIC,
    operating_profit NUMERIC,

    interest NUMERIC,
    interest_expense NUMERIC,
    finance_cost NUMERIC,

    other_income NUMERIC,
    non_operating_income NUMERIC,

    exceptional_items NUMERIC,
    extraordinary_items NUMERIC,

    pbt NUMERIC,
    profit_before_tax NUMERIC,

    tax NUMERIC,
    tax_expense NUMERIC,
    income_tax NUMERIC,

    share_of_profit_associates NUMERIC,
    minority_interest NUMERIC,

    reported_pat NUMERIC,
    net_profit NUMERIC,
    profit_after_tax NUMERIC,

    adj_pat NUMERIC,
    adjusted_pat NUMERIC,

    adjustments NUMERIC,

    eps NUMERIC,
    adj_eps NUMERIC,

    shares_outstanding NUMERIC,
    no_of_shares NUMERIC,

    -- QoQ and YoY growth
    revenue_yoy_growth_pct NUMERIC,
    revenue_qoq_growth_pct NUMERIC,
    ebitda_yoy_growth_pct NUMERIC,
    ebitda_qoq_growth_pct NUMERIC,
    pat_yoy_growth_pct NUMERIC,
    pat_qoq_growth_pct NUMERIC,

    ebitda_margin_pct NUMERIC,
    operating_margin_pct NUMERIC,
    net_margin_pct NUMERIC,

    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_profit_loss_quarterly_org ON profit_loss_quarterly(organization_id);
CREATE INDEX IF NOT EXISTS idx_profit_loss_quarterly_period ON profit_loss_quarterly(period);

-- Balance Sheet
CREATE TABLE IF NOT EXISTS balance_sheet (
    id BIGSERIAL PRIMARY KEY,
    organization_id UUID REFERENCES organization(id) ON DELETE CASCADE,
    fiscal_year TEXT NOT NULL,

    -- Current Assets
    cash NUMERIC,
    cash_equivalents NUMERIC,
    bank_balance NUMERIC,

    accounts_receivable NUMERIC,
    receivables NUMERIC,
    trade_receivables NUMERIC,
    debtors NUMERIC,

    inventories NUMERIC,
    stock NUMERIC,

    other_current_assets NUMERIC,
    prepaid_expenses NUMERIC,

    total_current_assets NUMERIC,

    -- Investments
    investments NUMERIC,
    long_term_investments NUMERIC,
    short_term_investments NUMERIC,

    -- Fixed Assets
    gross_fixed_assets NUMERIC,
    net_fixed_assets NUMERIC,
    ppe NUMERIC, -- Property, Plant & Equipment
    fixed_assets NUMERIC,

    cwip NUMERIC, -- Capital Work in Progress
    capital_wip NUMERIC,

    intangible_assets NUMERIC,
    goodwill NUMERIC,

    -- Other Assets
    deferred_tax_assets NUMERIC,
    other_non_current_assets NUMERIC,

    total_assets NUMERIC,

    -- Current Liabilities
    accounts_payable NUMERIC,
    payables NUMERIC,
    trade_payables NUMERIC,
    creditors NUMERIC,

    short_term_debt NUMERIC,
    current_portion_long_term_debt NUMERIC,

    other_current_liabilities NUMERIC,
    provisions NUMERIC,

    total_current_liabilities NUMERIC,

    -- Long-term Liabilities
    long_term_debt NUMERIC,
    bonds NUMERIC,

    deferred_tax_liabilities NUMERIC,
    other_non_current_liabilities NUMERIC,

    total_liabilities NUMERIC,

    -- Equity
    share_capital NUMERIC,
    equity_share_capital NUMERIC,

    reserves NUMERIC,
    reserves_surplus NUMERIC,
    retained_earnings NUMERIC,

    minority_interest NUMERIC,

    total_equity NUMERIC,
    shareholders_equity NUMERIC,
    net_worth NUMERIC,

    total_liabilities_equity NUMERIC,

    -- Calculated fields
    working_capital NUMERIC,
    net_debt NUMERIC,

    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_balance_sheet_org ON balance_sheet(organization_id);
CREATE INDEX IF NOT EXISTS idx_balance_sheet_fiscal_year ON balance_sheet(fiscal_year);

-- Change in Estimates
CREATE TABLE IF NOT EXISTS change_in_estimate (
    id BIGSERIAL PRIMARY KEY,
    organization_id UUID REFERENCES organization(id) ON DELETE CASCADE,
    fiscal_year TEXT NOT NULL,
    estimate_date DATE,

    -- Old estimates
    old_revenue NUMERIC,
    old_sales NUMERIC,
    old_ebitda NUMERIC,
    old_ebitda_margin_pct NUMERIC,
    old_margins_pct NUMERIC,
    old_pat NUMERIC,
    old_adj_pat NUMERIC,
    old_eps NUMERIC,

    -- New estimates
    new_revenue NUMERIC,
    new_sales NUMERIC,
    new_ebitda NUMERIC,
    new_ebitda_margin_pct NUMERIC,
    new_margins_pct NUMERIC,
    new_pat NUMERIC,
    new_adj_pat NUMERIC,
    new_eps NUMERIC,

    -- Change percentages
    revenue_change_pct NUMERIC,
    ebitda_change_pct NUMERIC,
    margins_change_bps NUMERIC, -- basis points
    pat_change_pct NUMERIC,
    eps_change_pct NUMERIC,

    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_change_estimate_org ON change_in_estimate(organization_id);
CREATE INDEX IF NOT EXISTS idx_change_estimate_fiscal_year ON change_in_estimate(fiscal_year);

-- Ratios
CREATE TABLE IF NOT EXISTS ratios (
    id BIGSERIAL PRIMARY KEY,
    organization_id UUID REFERENCES organization(id) ON DELETE CASCADE,
    fiscal_year TEXT NOT NULL,

    -- Profitability ratios
    roe NUMERIC, -- Return on Equity
    roa NUMERIC, -- Return on Assets
    roce NUMERIC, -- Return on Capital Employed
    roic NUMERIC, -- Return on Invested Capital

    -- Liquidity ratios
    current_ratio NUMERIC,
    quick_ratio NUMERIC,
    cash_ratio NUMERIC,

    -- Leverage ratios
    debt_to_equity NUMERIC,
    debt_to_assets NUMERIC,
    interest_coverage NUMERIC,

    -- Efficiency ratios
    asset_turnover NUMERIC,
    inventory_turnover NUMERIC,
    receivables_turnover NUMERIC,
    payables_turnover NUMERIC,

    -- Valuation ratios
    pe_ratio NUMERIC,
    pb_ratio NUMERIC,
    ps_ratio NUMERIC,
    ev_ebitda NUMERIC,

    -- Other
    dividend_yield NUMERIC,
    payout_ratio NUMERIC,

    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ratios_org ON ratios(organization_id);
CREATE INDEX IF NOT EXISTS idx_ratios_fiscal_year ON ratios(fiscal_year);

-- Cash Flow
CREATE TABLE IF NOT EXISTS cash_flow (
    id BIGSERIAL PRIMARY KEY,
    organization_id UUID REFERENCES organization(id) ON DELETE CASCADE,
    fiscal_year TEXT NOT NULL,

    -- Operating activities
    cash_from_operations NUMERIC,
    operating_cash_flow NUMERIC,
    cfo NUMERIC,

    -- Investing activities
    cash_from_investing NUMERIC,
    investing_cash_flow NUMERIC,
    cfi NUMERIC,
    capex NUMERIC,
    capital_expenditure NUMERIC,

    -- Financing activities
    cash_from_financing NUMERIC,
    financing_cash_flow NUMERIC,
    cff NUMERIC,

    -- Net change
    net_cash_flow NUMERIC,
    free_cash_flow NUMERIC,
    fcf NUMERIC,

    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_cash_flow_org ON cash_flow(organization_id);
CREATE INDEX IF NOT EXISTS idx_cash_flow_fiscal_year ON cash_flow(fiscal_year);

-- Company Data (general company information)
CREATE TABLE IF NOT EXISTS company_data (
    id BIGSERIAL PRIMARY KEY,
    organization_id UUID REFERENCES organization(id) ON DELETE CASCADE,

    company_name TEXT,
    ticker_symbol TEXT,
    exchange TEXT,
    isin TEXT,

    ceo TEXT,
    cfo TEXT,
    founded_date DATE,

    headquarters TEXT,
    website TEXT,

    employees INT,

    description TEXT,
    business_segments TEXT[],
    key_products TEXT[],

    market_cap NUMERIC,
    enterprise_value NUMERIC,

    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_company_data_org ON company_data(organization_id);

-- Shareholding Percentage
CREATE TABLE IF NOT EXISTS shareholding_percentage (
    id BIGSERIAL PRIMARY KEY,
    organization_id UUID REFERENCES organization(id) ON DELETE CASCADE,
    as_of_date DATE NOT NULL,

    promoter_holding_pct NUMERIC,
    public_holding_pct NUMERIC,
    institutional_holding_pct NUMERIC,

    fii_holding_pct NUMERIC,
    dii_holding_pct NUMERIC,
    retail_holding_pct NUMERIC,

    pledged_shares_pct NUMERIC,

    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_shareholding_org ON shareholding_percentage(organization_id);
CREATE INDEX IF NOT EXISTS idx_shareholding_date ON shareholding_percentage(as_of_date);

-- Price Performance
CREATE TABLE IF NOT EXISTS price_performance (
    id BIGSERIAL PRIMARY KEY,
    organization_id UUID REFERENCES organization(id) ON DELETE CASCADE,
    date DATE NOT NULL,

    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC,

    volume BIGINT,

    -- Returns
    return_1d_pct NUMERIC,
    return_1w_pct NUMERIC,
    return_1m_pct NUMERIC,
    return_3m_pct NUMERIC,
    return_6m_pct NUMERIC,
    return_1y_pct NUMERIC,
    return_3y_pct NUMERIC,
    return_5y_pct NUMERIC,

    -- Peak/trough
    high_52w NUMERIC,
    low_52w NUMERIC,

    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_price_performance_org ON price_performance(organization_id);
CREATE INDEX IF NOT EXISTS idx_price_performance_date ON price_performance(date);
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def upsert_document(conn: psycopg.Connection, path: str, sha256: str, title: str | None, pages: int | None) -> int:
    sql = (
        "INSERT INTO documents(path, sha256, title, pages) VALUES(%s,%s,%s,%s) "
        "ON CONFLICT(path) DO UPDATE SET sha256=EXCLUDED.sha256, title=EXCLUDED.title, pages=EXCLUDED.pages "
        "RETURNING id"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (path, sha256, title, pages))
        (doc_id,) = cur.fetchone()
    conn.commit()
    return doc_id


def insert_chunks(
    conn: psycopg.Connection,
    document_id: int,
    chunks: list[tuple[int, str, int | None, int | None, int | None]],
) -> None:
    if not chunks:
        return
    sql = (
        "INSERT INTO chunks(document_id, chunk_index, text, page_start, page_end, token_count) "
        "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(document_id, chunk_index) DO NOTHING"
    )
    with conn.cursor() as cur:
        cur.executemany(sql, [(document_id, i, t, ps, pe, tok) for i, t, ps, pe, tok in chunks])
    conn.commit()


def upsert_company(conn: psycopg.Connection, name: str, sector: str | None = None, industry: str | None = None, metadata: dict | None = None) -> int:
    """Insert or update a company and return its ID."""
    import json
    sql = """
        INSERT INTO companies(name, sector, industry, metadata, updated_at)
        VALUES(%s, %s, %s, %s, now())
        ON CONFLICT(name) DO UPDATE SET
            sector = COALESCE(EXCLUDED.sector, companies.sector),
            industry = COALESCE(EXCLUDED.industry, companies.industry),
            metadata = companies.metadata || EXCLUDED.metadata,
            updated_at = now()
        RETURNING id
    """
    with conn.cursor() as cur:
        cur.execute(sql, (name, sector, industry, json.dumps(metadata or {})))
        (company_id,) = cur.fetchone()
    conn.commit()
    return company_id


def insert_financial_metric(
    conn: psycopg.Connection,
    company_id: int,
    document_id: int,
    metric_name: str,
    metric_value: float | None = None,
    metric_value_text: str | None = None,
    unit: str | None = None,
    period: str | None = None,
    quarter: int | None = None,
    fiscal_year: int | None = None,
    report_date: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Insert a financial metric for a company."""
    import json
    sql = """
        INSERT INTO financial_metrics(
            company_id, document_id, metric_name, metric_value, metric_value_text,
            unit, period, quarter, fiscal_year, report_date, metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (
            company_id, document_id, metric_name, metric_value, metric_value_text,
            unit, period, quarter, fiscal_year, report_date, json.dumps(metadata or {})
        ))
    conn.commit()


def upsert_organization(
    conn: psycopg.Connection,
    name: str,
    company_id: int | None = None,
    document_id: int | None = None,
    sector: str | None = None,
    industry: str | None = None,
    pdf_path: str | None = None,
    parsing_method: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Insert or update an organization and return its UUID."""
    import json
    sql = """
        INSERT INTO organization(name, company_id, document_id, sector, industry, pdf_path, parsing_method, metadata, updated_at)
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            company_id = COALESCE(EXCLUDED.company_id, organization.company_id),
            document_id = COALESCE(EXCLUDED.document_id, organization.document_id),
            sector = COALESCE(EXCLUDED.sector, organization.sector),
            industry = COALESCE(EXCLUDED.industry, organization.industry),
            pdf_path = COALESCE(EXCLUDED.pdf_path, organization.pdf_path),
            parsing_method = COALESCE(EXCLUDED.parsing_method, organization.parsing_method),
            metadata = organization.metadata || EXCLUDED.metadata,
            updated_at = now()
        RETURNING id
    """
    with conn.cursor() as cur:
        cur.execute(sql, (name, company_id, document_id, sector, industry, pdf_path, parsing_method, json.dumps(metadata or {})))
        (org_id,) = cur.fetchone()
    conn.commit()
    return str(org_id)


def get_organization_by_name(conn: psycopg.Connection, name: str) -> str | None:
    """Get organization UUID by name."""
    sql = "SELECT id FROM organization WHERE LOWER(name) = LOWER(%s) LIMIT 1"
    with conn.cursor() as cur:
        cur.execute(sql, (name,))
        row = cur.fetchone()
        return str(row[0]) if row else None


def insert_profit_loss(conn: psycopg.Connection, organization_id: str, fiscal_year: str, data: dict) -> None:
    """Insert profit & loss data for an organization."""
    import json
    columns = ', '.join(data.keys())
    placeholders = ', '.join(['%s'] * len(data))
    sql = f"""
        INSERT INTO profit_loss(organization_id, fiscal_year, {columns}, metadata)
        VALUES (%s, %s, {placeholders}, %s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (organization_id, fiscal_year, *data.values(), json.dumps({})))
    conn.commit()


def insert_profit_loss_quarterly(conn: psycopg.Connection, organization_id: str, period: str, data: dict) -> None:
    """Insert quarterly profit & loss data for an organization."""
    import json
    columns = ', '.join(data.keys())
    placeholders = ', '.join(['%s'] * len(data))
    sql = f"""
        INSERT INTO profit_loss_quarterly(organization_id, period, {columns}, metadata)
        VALUES (%s, %s, {placeholders}, %s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (organization_id, period, *data.values(), json.dumps({})))
    conn.commit()


def insert_balance_sheet(conn: psycopg.Connection, organization_id: str, fiscal_year: str, data: dict) -> None:
    """Insert balance sheet data for an organization."""
    import json
    columns = ', '.join(data.keys())
    placeholders = ', '.join(['%s'] * len(data))
    sql = f"""
        INSERT INTO balance_sheet(organization_id, fiscal_year, {columns}, metadata)
        VALUES (%s, %s, {placeholders}, %s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (organization_id, fiscal_year, *data.values(), json.dumps({})))
    conn.commit()


def insert_change_in_estimate(conn: psycopg.Connection, organization_id: str, fiscal_year: str, data: dict) -> None:
    """Insert change in estimate data for an organization."""
    import json
    columns = ', '.join(data.keys())
    placeholders = ', '.join(['%s'] * len(data))
    sql = f"""
        INSERT INTO change_in_estimate(organization_id, fiscal_year, {columns}, metadata)
        VALUES (%s, %s, {placeholders}, %s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (organization_id, fiscal_year, *data.values(), json.dumps({})))
    conn.commit()


def get_schema_json() -> dict:
    """Return a JSON representation of the database schema for LLM context."""
    return {
        "organization": {
            "description": "Main table for organizations/companies",
            "columns": ["id (UUID)", "name", "sector", "industry", "pdf_path", "parsing_method", "metadata"],
            "relationships": ["company_id -> companies.id", "document_id -> documents.id"]
        },
        "profit_loss": {
            "description": "Annual profit & loss statements",
            "columns": [
                "organization_id (UUID FK)", "fiscal_year",
                "revenue/sales/total_income/turnover",
                "ebitda/operating_profit_before_depreciation",
                "depreciation/amortization",
                "ebit/operating_profit",
                "interest/interest_expense/finance_cost",
                "other_income/non_operating_income",
                "exceptional_items/extraordinary_items",
                "pbt/profit_before_tax",
                "tax/tax_expense/income_tax/tax_rate",
                "reported_pat/net_profit/profit_after_tax",
                "adj_pat/adjusted_pat/normalized_pat",
                "eps/adj_eps/basic_eps/diluted_eps",
                "shares_outstanding/no_of_shares",
                "*_growth_pct fields",
                "*_margin_pct fields"
            ],
            "note": "Multiple column names for same metric (different company naming conventions)"
        },
        "profit_loss_quarterly": {
            "description": "Quarterly profit & loss statements",
            "columns": ["Same as profit_loss", "period (e.g., Q4FY24)", "quarter", "fiscal_year", "yoy_growth_pct", "qoq_growth_pct"],
            "note": "Use for quarterly analysis and comparisons"
        },
        "balance_sheet": {
            "description": "Balance sheet data",
            "columns": [
                "organization_id (UUID FK)", "fiscal_year",
                "cash/cash_equivalents/bank_balance",
                "accounts_receivable/receivables/trade_receivables/debtors",
                "inventories/stock",
                "other_current_assets",
                "investments",
                "gross_fixed_assets/net_fixed_assets/ppe",
                "cwip/capital_wip",
                "intangible_assets/goodwill",
                "total_assets",
                "accounts_payable/payables/trade_payables/creditors",
                "short_term_debt",
                "long_term_debt",
                "total_liabilities",
                "share_capital/equity_share_capital",
                "reserves/reserves_surplus/retained_earnings",
                "total_equity/shareholders_equity/net_worth",
                "working_capital",
                "net_debt"
            ]
        },
        "change_in_estimate": {
            "description": "Changes in financial estimates over time",
            "columns": [
                "organization_id (UUID FK)", "fiscal_year", "estimate_date",
                "old_* and new_* versions of: revenue, sales, ebitda, margins, pat, eps",
                "*_change_pct fields"
            ]
        },
        "ratios": {
            "description": "Financial ratios",
            "columns": [
                "organization_id (UUID FK)", "fiscal_year",
                "roe/roa/roce/roic (profitability)",
                "current_ratio/quick_ratio/cash_ratio (liquidity)",
                "debt_to_equity/debt_to_assets/interest_coverage (leverage)",
                "asset_turnover/inventory_turnover/*_turnover (efficiency)",
                "pe_ratio/pb_ratio/ps_ratio/ev_ebitda (valuation)",
                "dividend_yield/payout_ratio"
            ]
        },
        "cash_flow": {
            "description": "Cash flow statements",
            "columns": [
                "organization_id (UUID FK)", "fiscal_year",
                "cash_from_operations/operating_cash_flow/cfo",
                "cash_from_investing/investing_cash_flow/cfi",
                "capex/capital_expenditure",
                "cash_from_financing/financing_cash_flow/cff",
                "net_cash_flow",
                "free_cash_flow/fcf"
            ]
        },
        "company_data": {
            "description": "General company information",
            "columns": [
                "organization_id (UUID FK)",
                "company_name", "ticker_symbol", "exchange", "isin",
                "ceo", "cfo", "founded_date",
                "headquarters", "website", "employees",
                "description", "business_segments", "key_products",
                "market_cap", "enterprise_value"
            ]
        },
        "shareholding_percentage": {
            "description": "Shareholding patterns",
            "columns": [
                "organization_id (UUID FK)", "as_of_date",
                "promoter_holding_pct", "public_holding_pct", "institutional_holding_pct",
                "fii_holding_pct", "dii_holding_pct", "retail_holding_pct",
                "pledged_shares_pct"
            ]
        },
        "price_performance": {
            "description": "Stock price and returns",
            "columns": [
                "organization_id (UUID FK)", "date",
                "open_price", "high_price", "low_price", "close_price", "volume",
                "return_1d_pct", "return_1w_pct", "return_1m_pct", "return_3m_pct",
                "return_6m_pct", "return_1y_pct", "return_3y_pct", "return_5y_pct",
                "high_52w", "low_52w"
            ]
        },
        "companies": {
            "description": "Legacy company table (still used for metadata)",
            "columns": ["id", "name", "sector", "industry", "metadata"]
        },
        "financial_metrics": {
            "description": "Generic metrics table (flexible key-value storage)",
            "columns": ["company_id", "document_id", "metric_name", "metric_value", "period", "fiscal_year"]
        },
        "documents": {
            "description": "Ingested PDF documents",
            "columns": ["id", "path", "sha256", "title", "pages"]
        },
        "chunks": {
            "description": "Vector-searchable text chunks from documents",
            "columns": ["document_id", "chunk_index", "text", "page_start", "page_end"]
        }
    }

