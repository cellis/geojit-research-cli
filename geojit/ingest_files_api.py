"""Ingestion pipeline using OpenAI File Inputs (Responses API).

This pipeline sends whole PDF files to GPT-5 via the Responses API using one of
three methods: upload (Files API -> file_id), external URL (file_url), or
base64-encoded file data. The model receives both extracted PDF text and page
images, improving table extraction accuracy without us rendering pages.

For each PDF, we request structured JSON for:
- profit_loss_quarterly
- profit_loss (annual)
- balance_sheet
- change_in_estimate
- ratios
- cash_flow

We then optionally insert into Postgres and index text chunks into Qdrant.

CLI:
  python -m geojit.ingest_files_api --file Financial_Research_Agent_Files/SP20241406115209223TTK.pdf --method upload --eval --eval-csv "SP20241406115209223TTK - profit_loss.csv" --no-db

  python -m geojit.ingest_files_api --max-files 10 --method upload

Notes:
- Requires OPENAI_API_KEY.
- For method=url, provide --file-url instead of --file.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

from .config import load_settings
from .db import (
    connect,
    ensure_schema,
    upsert_document,
    upsert_company,
    upsert_organization,
)
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


def _build_system_prompt(company_hint: str | None) -> str:
    return f"""You are a financial data extraction expert.
Analyze the provided PDF and extract STRICT JSON with the schema below. Use null for missing values. No commentary.

Company hint: {company_hint or 'Unknown'}

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
  }}]
}}

INSTRUCTIONS:
1) Extract ALL years/quarters; use null for missing.
2) Convert percentages to numeric percent units (12.4% -> 12.4).
3) Remove thousands separators (2,678 -> 2678).
4) Return ONLY JSON (no code fences).
5) If a section is not present, return an empty array for that section.
"""


def _responses_text(resp: Any) -> str:
    """Extract text from Responses API response object across SDK variants."""
    # Try output_text first (newer SDKs)
    txt = getattr(resp, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    # Fallback to content array
    try:
        data = resp.to_dict() if hasattr(resp, "to_dict") else resp
        # Look for top-level output > content (list)
        out = data.get("output") if isinstance(data, dict) else None
        if isinstance(out, list):
            parts = []
            for item in out:
                if isinstance(item, dict):
                    content = item.get("content")
                    if isinstance(content, list):
                        for c in content:
                            if c.get("type") in {"output_text", "text"} and c.get("text"):
                                parts.append(c["text"])
            if parts:
                return "\n".join(parts).strip()
    except Exception:
        pass
    # Last resort: look for choices like Chat API
    try:
        c0 = resp.choices[0].message.content
        if isinstance(c0, str):
            return c0.strip()
    except Exception:
        pass
    raise ValueError("Could not extract text from Responses API result")


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


def _parse_with_files_api(client: OpenAI, pdf_path: Optional[Path] = None, *, method: str, file_url: Optional[str] = None, company_hint: Optional[str] = None) -> dict:
    """Send a PDF using OpenAI File Inputs and parse structured tables with GPT-5.

    method: 'upload' | 'url' | 'base64'
    If method='url', provide file_url.
    If method!='url', provide pdf_path.
    """
    content = []
    content.append({"type": "input_text", "text": "Extract structured financial tables as strict JSON using the schema in the system prompt."})

    if method == "url":
        if not file_url:
            raise ValueError("file_url required for method=url")
        content.append({"type": "input_file", "file_url": file_url})
    elif method == "upload":
        assert pdf_path is not None
        # Upload to Files API
        with open(pdf_path, "rb") as f:
            fobj = client.files.create(file=f, purpose="user_data")
        content.append({"type": "input_file", "file_id": fobj.id})
    elif method == "base64":
        assert pdf_path is not None
        b = pdf_path.read_bytes()
        b64 = base64.b64encode(b).decode("utf-8")
        content.append({"type": "input_file", "filename": pdf_path.name, "file_data": b64})
    else:
        raise ValueError("method must be one of: upload, url, base64")

    system_prompt = _build_system_prompt(company_hint)

    resp = client.responses.create(
        model="gpt-5",
        input=[{"role": "user", "content": content}],
        system=system_prompt,
        max_output_tokens=16000,
    )

    text = _responses_text(resp)
    text = _strip_code_fences(text)
    data = json.loads(text)
    return data


def ingest_files_api(max_files: int | None = None, *, file: str | None = None, method: str = "upload", file_url: str | None = None, no_db: bool = False) -> None:
    s = load_settings()
    client = OpenAI(api_key=s.openai_api_key)

    if file:
        pdfs = [Path(file)]
    elif method == "url" and file_url:
        # We still need a placeholder Path for metadata; use URL stem
        pdfs = [Path(file_url)]
    else:
        root = s.data_dir
        pdfs = sorted(list(root.glob("**/*.pdf")))
        if max_files is not None:
            pdfs = pdfs[:max_files]

    if not pdfs:
        print("No PDFs found")
        return

    conn = connect(s.database_url)
    ensure_schema(conn)
    qdrant = get_qdrant(s.qdrant_url, s.qdrant_api_key)
    main_collection = s.qdrant_collection
    main_vector_size: int | None = None

    for idx, p in enumerate(pdfs, 1):
        try:
            print(f"[{idx}/{len(pdfs)}] {p.name}")

            # Metadata + text for chunking
            if method == "url" and file_url:
                sha = hashlib.sha256(file_url.encode("utf-8")).hexdigest()
                title_hint = p.stem
                doc_id = upsert_document(conn, file_url, sha, title_hint, None)
                company_hint = p.stem
            else:
                sha = _sha256_file(p)
                doc = extract_pdf(p)
                doc_id = upsert_document(conn, str(p), sha, doc.title, doc.pages)
                company_hint = p.stem

                # Chunk + embed (fast path)
                chunks = chunk_pages(doc.texts, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
                insert_chunks(
                    conn,
                    document_id=doc_id,
                    chunks=[(c.index, c.text, c.page_start, c.page_end, c.token_count) for c in chunks],
                )
                try:
                    batch = 128
                    for i in range(0, len(chunks), batch):
                        slice_chunks = chunks[i : i + batch]
                        texts = [c.text for c in slice_chunks]
                        vecs = embed_texts(texts, api_key=s.openai_api_key, model=s.embedding_model)
                        if main_vector_size is None and vecs.shape[0] > 0:
                            main_vector_size = vecs.shape[1]
                            ensure_collection(qdrant, main_collection, main_vector_size)
                        ids = [f"{doc_id}:{c.index}" for c in slice_chunks]
                        payloads = [
                            {
                                "type": "chunk",
                                "document_id": doc_id,
                                "path": str(p),
                                "chunk_index": c.index,
                                "page_start": c.page_start,
                                "page_end": c.page_end,
                                "title": doc.title if method != "url" else None,
                            }
                            for c in slice_chunks
                        ]
                        upsert_points(qdrant, main_collection, ids, vecs.tolist(), payloads)
                except Exception as e:
                    print(f"  ! Skipped Qdrant chunk embeddings: {e}")

            # Parse structured tables via File Inputs
            parsed = _parse_with_files_api(client, p if method != "url" else None, method=method, file_url=file_url, company_hint=company_hint)

            # Optionally insert into DB
            if not no_db:
                company_name = parsed.get("company_name", company_hint)
                company_id = upsert_company(conn, name=company_name, sector=parsed.get("sector"), industry=parsed.get("industry"), metadata={})
                org_id = upsert_organization(conn, name=company_name, company_id=company_id, document_id=doc_id, sector=parsed.get("sector"), industry=parsed.get("industry"), pdf_path=str(p if method != "url" else file_url), parsing_method="file-inputs", metadata={})

                for item in parsed.get("profit_loss_quarterly", []) or []:
                    _insert_profit_loss_quarterly(conn, org_id, item)
                for item in parsed.get("profit_loss", []) or []:
                    _insert_profit_loss(conn, org_id, item)
                for item in parsed.get("balance_sheet", []) or []:
                    _insert_balance_sheet(conn, org_id, item)
                for item in parsed.get("change_in_estimate", []) or []:
                    _insert_change_in_estimate(conn, org_id, item)
                for item in parsed.get("ratios", []) or []:
                    _insert_ratios(conn, org_id, item)
                for item in parsed.get("cash_flow", []) or []:
                    _insert_cash_flow(conn, org_id, item)

            # Dump parsed JSON for inspection
            out_json = Path.cwd() / f"parsed_output_filesapi_{p.stem}.json"
            try:
                with open(out_json, "w") as f:
                    json.dump(parsed, f, indent=2, default=str)
            except Exception:
                pass

            print("  ✓ Parsed (Files API)")
        except Exception as e:
            print(f"  ✗ Failed {p.name}: {e}")


def eval_files_api(pdf_path: str, csv_paths: list[str], *, method: str = "upload", file_url: str | None = None) -> dict[str, Any]:
    s = load_settings()
    client = OpenAI(api_key=s.openai_api_key)
    p = Path(pdf_path)
    company_hint = p.stem
    parsed = _parse_with_files_api(client, p if method != "url" else None, method=method, file_url=file_url, company_hint=company_hint)

    # Save parsed JSON
    out_json = Path.cwd() / f"parsed_output_filesapi_{p.stem}.json"
    with open(out_json, "w") as f:
        json.dump(parsed, f, indent=2, default=str)

    results = evaluate_parser_output(parsed, csv_paths)
    print_evaluation_report(results)
    eval_out = Path.cwd() / f"eval_results_filesapi_{p.stem}.json"
    with open(eval_out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="ingest_files_api", description="Ingest PDFs using OpenAI File Inputs (Responses API)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of PDFs to process")
    parser.add_argument("--file", type=str, default=None, help="Path to a single local PDF to process")
    parser.add_argument("--method", type=str, default="upload", choices=["upload", "url", "base64"], help="How to send the file to the API")
    parser.add_argument("--file-url", type=str, default=None, help="External URL when --method=url")
    parser.add_argument("--no-db", action="store_true", help="Do not write to DB; only parse and write JSON + index chunks")
    parser.add_argument("--eval", action="store_true", help="Run evaluation against CSV ground truth")
    parser.add_argument("--eval-csv", action="append", default=None, help="Path to a CSV file (repeatable)")
    args = parser.parse_args()

    if args.eval:
        if not args.file and args.method != "url":
            raise SystemExit("--eval requires --file unless --method=url with --file-url")
        if not args.eval_csv:
            raise SystemExit("--eval requires at least one --eval-csv path")
        eval_files_api(args.file or "url", args.eval_csv, method=args.method, file_url=args.file_url)
    else:
        ingest_files_api(max_files=args.max_files, file=args.file, method=args.method, file_url=args.file_url, no_db=args.no_db)

