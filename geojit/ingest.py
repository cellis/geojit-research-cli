from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Any
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import load_settings
from .db import (
    connect, ensure_schema, upsert_document, insert_chunks,
    upsert_company, insert_financial_metric, upsert_organization,
    get_organization_by_name
)
from .pdf_parser import extract_pdf
from .pdf_parser_text import parse_pdf_text_based
from .chunking import chunk_pages
from .embeddings import embed_texts
from .qdrant_store import get_qdrant, ensure_collection, upsert_points
from .metadata_extractor import extract_document_metadata
from .pdf_evaluator import evaluate_parser_output, print_evaluation_report


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


def _insert_profit_loss_quarterly(conn, org_id: str, data: dict) -> None:
    """Insert quarterly P&L data."""
    import json
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


def _insert_ratios(conn, org_id: str, data: dict) -> None:
    """Insert ratios data."""
    fields = {k: v for k, v in data.items() if k != 'fiscal_year' and v is not None}
    if not fields:
        return

    columns = ', '.join(['organization_id', 'fiscal_year'] + list(fields.keys()))
    placeholders = ', '.join(['%s'] * (2 + len(fields)))
    sql = f"INSERT INTO ratios({columns}) VALUES ({placeholders})"

    with conn.cursor() as cur:
        cur.execute(sql, (org_id, data.get('fiscal_year'), *fields.values()))
    conn.commit()


def _insert_cash_flow(conn, org_id: str, data: dict) -> None:
    """Insert cash flow data."""
    fields = {k: v for k, v in data.items() if k != 'fiscal_year' and v is not None}
    if not fields:
        return

    columns = ', '.join(['organization_id', 'fiscal_year'] + list(fields.keys()))
    placeholders = ', '.join(['%s'] * (2 + len(fields)))
    sql = f"INSERT INTO cash_flow({columns}) VALUES ({placeholders})"

    with conn.cursor() as cur:
        cur.execute(sql, (org_id, data.get('fiscal_year'), *fields.values()))
    conn.commit()


def _process_one(path: Path, s, qdrant, vector_size_box: dict, *, skip_parse: bool = False) -> tuple[str, bool, str | None]:
    """Process a single PDF. Returns (name, success, error)."""
    try:
        sha = _sha256_file(path)

        conn = connect(s.database_url)
        # assume schema ensured by caller

        # Dedup per-file (lightweight query)
        with conn.cursor() as cur:
            cur.execute("SELECT id, sha256 FROM documents WHERE path = %s", (str(path),))
            existing = cur.fetchone()
            if existing:
                existing_id, existing_sha = existing
                if existing_sha == sha:
                    return (path.name, True, "skipped")

        doc = extract_pdf(path)
        doc_id = upsert_document(conn, str(path), sha, doc.title, doc.pages)
        chunks = chunk_pages(doc.texts, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)

        # Text-based parse (can be slow; keep but allow parallelism across files)
        company_hint = path.stem.split('_')[0] if '_' in path.stem else path.stem
        parsed_data = None
        if not skip_parse:
            try:
                parsed_data = parse_pdf_text_based(str(path), company_hint=company_hint)
                company_name = parsed_data.get('company_name', company_hint)
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
                for item in parsed_data.get('profit_loss_quarterly', []):
                    _insert_profit_loss_quarterly(conn, org_id, item)
                for item in parsed_data.get('profit_loss', []):
                    _insert_profit_loss(conn, org_id, item)
                for item in parsed_data.get('balance_sheet', []):
                    _insert_balance_sheet(conn, org_id, item)
                for item in parsed_data.get('change_in_estimate', []):
                    _insert_change_in_estimate(conn, org_id, item)
                for item in parsed_data.get('ratios', []):
                    _insert_ratios(conn, org_id, item)
                for item in parsed_data.get('cash_flow', []):
                    _insert_cash_flow(conn, org_id, item)
            except Exception:
                # Parsing is optional; continue with chunking and embeddings
                pass

        # Insert chunks
        insert_chunks(
            conn,
            document_id=doc_id,
            chunks=[(c.index, c.text, c.page_start, c.page_end, c.token_count) for c in chunks],
        )

        # Embeddings in micro-batches; share collection creation via vector_size_box
        batch = 128
        for i in range(0, len(chunks), batch):
            slice_chunks = chunks[i : i + batch]
            texts = [c.text for c in slice_chunks]
            vecs = embed_texts(texts, api_key=s.openai_api_key, model=s.embedding_model)
            if vector_size_box.get('size') is None and vecs.shape[0] > 0:
                vector_size_box['size'] = vecs.shape[1]
                ensure_collection(qdrant, s.qdrant_collection, vector_size_box['size'])
            ids = [f"{doc_id}:{c.index}" for c in slice_chunks]
            payloads = [
                {
                    "document_id": doc_id,
                    "path": str(path),
                    "chunk_index": c.index,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "title": doc.title,
                }
                for c in slice_chunks
            ]
            upsert_points(qdrant, s.qdrant_collection, ids, vecs.tolist(), payloads)

        return (path.name, True, None)
    except Exception as e:
        return (path.name, False, str(e))


def ingest_all(max_files: int | None = None, *, workers: int = 4, skip_parse: bool = False) -> None:
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
    vector_size_box: dict = {"size": None}

    # Ensure schema once up front
    conn = connect(s.database_url)
    ensure_schema(conn)
    conn.close()

    # Threaded ingestion across files. Each worker handles its own DB session.
    workers = max(1, int(os.getenv("GEOJIT_INGEST_WORKERS", str(workers))))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_one, p, s, qdrant, vector_size_box, skip_parse=skip_parse): p for p in pdfs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Ingest PDFs (workers={workers})"):
            name, ok, err = fut.result()
            if err == "skipped":
                print(f"  ⊘ Skipped {name}")
            elif not ok:
                print(f"  ✗ Failed {name}: {err}")


def eval_ingest(pdf_path: str, csv_paths: list[str]) -> dict[str, Any]:
    """Run text-based extraction for a single PDF and evaluate against CSVs."""
    p = Path(pdf_path)

    # Parse PDF using text-based parser (which handles extraction internally)
    parsed = parse_pdf_text_based(str(p), company_hint=p.stem)

    # Save for inspection
    out_json = Path.cwd() / f"parsed_output_text_{p.stem}.json"
    with open(out_json, 'w') as f:
        json.dump(parsed, f, indent=2, default=str)

    # Evaluate
    results = evaluate_parser_output(parsed, csv_paths)
    print_evaluation_report(results)

    # Save evaluation results
    eval_out = Path.cwd() / f"eval_results_text_{p.stem}.json"
    with open(eval_out, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="ingest", description="Ingest PDFs using text-based extraction")
    parser.add_argument("--file", type=str, default=None, help="Path to a single PDF to process")
    parser.add_argument("--eval", action="store_true", help="Run evaluation against CSV ground truth")
    parser.add_argument("--eval-csv", action="append", default=None, help="Path to a CSV file (repeatable)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers for ingestion")
    parser.add_argument("--skip-parse", action="store_true", help="Skip LLM text parsing (embed-only for speed)")
    parser.add_argument("--db-url", type=str, default=None, help="Override DATABASE_URL for this run")
    args = parser.parse_args()

    if args.eval:
        if not args.file:
            raise SystemExit("--eval requires --file to be specified")
        if not args.eval_csv:
            raise SystemExit("--eval requires at least one --eval-csv path")
        eval_ingest(args.file, args.eval_csv)
    else:
        # Allow per-run DB override
        if args.db_url:
            import os as _os
            _os.environ["DATABASE_URL"] = args.db_url
        # Normal ingestion mode
        if args.file:
            # Note: current implementation still processes by directory; use max_files=1 as a convenience
            ingest_all(max_files=1, workers=args.workers, skip_parse=args.skip_parse)
        else:
            ingest_all(workers=args.workers, skip_parse=args.skip_parse)
