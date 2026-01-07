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

