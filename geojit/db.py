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

