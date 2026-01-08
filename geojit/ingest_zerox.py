"""Zerox-enabled ingestion agent for extracting financial tables and inserting into DB.

This agent scans PDFs under the configured data directory, extracts the following
tables for each company, and inserts them into the `geojit` database schema:

- organization
- profit_loss
- profit_loss_quarterly
- balance_sheet
- change_in_estimate
- ratios
- cash_flow
- shareholding_percentage
- price_performance

Primary extraction path is Python + GPT-5 (ai-sdk) over text content. Optionally,
this agent can shell out to a Node.js runner that uses the `zerox` library
(`npm install zerox`) if available. If the Node runner is not available or fails,
it falls back to the Python text-based extraction.

All logic resides in this file; any Node runner script is written to a temporary
directory at runtime (no extra repo files).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import load_settings
from .db import (
    connect,
    ensure_schema,
    upsert_document,
    upsert_company,
    upsert_organization,
)

# Reuse helper insert functions for core sections from geojit.ingest
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


# Optional: ai-sdk for GPT-5 parsing fallback
try:
    from ai_sdk import generate_text, openai
    from ai_sdk.types import CoreSystemMessage, CoreUserMessage, TextPart
    _ai_sdk_available = True
except Exception:
    _ai_sdk_available = False

# Optional: pypdf text extraction fallback
try:
    import PyPDF2
    _pypdf2_available = True
except Exception:
    _pypdf2_available = False


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


# --------------------------------------------------------------------------------------
# Optional Zerox (Node.js) Runner
# --------------------------------------------------------------------------------------

ZER0X_RUNNER_JS = r"#!/usr/bin/env node\n"
ZER0X_RUNNER_JS += r"""
// Minimal Node.js runner that attempts to use `zerox` to extract structured tables
// from text content passed via STDIN or directly from a PDF path.
//
// Usage:
//   node index.mjs --pdf "/path/to/file.pdf"
// or
//   echo '{"text": "..."}' | node index.mjs
//
// The script outputs a JSON object with keys matching the expected schema:
// company_name, sector, industry, profit_loss, profit_loss_quarterly,
// balance_sheet, change_in_estimate, ratios, cash_flow, shareholding_percentage,
// price_performance.

import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

// Try to resolve zerox relative to a base directory (repo root)
const base = process.env.ZEROX_BASE || process.cwd();
const requireFromBase = createRequire(base.endsWith('/') ? base : base + '/');

let runZerox = null;
try {
  const mod = requireFromBase('zerox');
  runZerox = (mod && mod.zerox) ? mod.zerox : (mod && mod.default ? mod.default.zerox : null);
} catch (e1) {
  try {
    const dyn = await import('zerox');
    runZerox = (dyn && dyn.zerox) ? dyn.zerox : (dyn && dyn.default ? dyn.default.zerox : null);
  } catch (e2) {
    console.error(JSON.stringify({ error: 'ZER0X_NOT_INSTALLED', message: String(e2) }));
    process.exit(2);
  }
}

// Parse args
const args = process.argv.slice(2);
const argMap = new Map();
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a.startsWith('--')) {
    const key = a.replace(/^--/, '');
    const val = args[i + 1] && !args[i + 1].startsWith('--') ? args[++i] : true;
    argMap.set(key, val);
  }
}

async function readStdin() {
  return new Promise((resolve) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => (data += chunk));
    process.stdin.on('end', () => resolve(data));
  });
}

async function main() {
  let inputText = null;
  if (process.stdin.isTTY === false) {
    const raw = await readStdin();
    try {
      const obj = JSON.parse(raw);
      if (obj && typeof obj.text === 'string') {
        inputText = obj.text;
      }
    } catch {}
  }

  const pdfPath = argMap.get('pdf');

  // Build a prompt describing our desired JSON schema
  const prompt = `Extract the following financial tables and output JSON only:\n\n{
  "company_name": "string",
  "sector": "string|null",
  "industry": "string|null",
  "profit_loss_quarterly": [ { "period": "Q4FY24", "quarter": 4, "fiscal_year": 2024, "sales": number, "revenue": number, "ebitda": number, "ebitda_margin_pct": number, "depreciation": number, "ebit": number, "interest": number, "other_income": number, "exceptional_items": number, "pbt": number, "tax": number, "reported_pat": number, "adj_pat": number, "eps": number, "adj_eps": number, "shares_outstanding": number, "revenue_yoy_growth_pct": number, "revenue_qoq_growth_pct": number, "ebitda_yoy_growth_pct": number, "ebitda_qoq_growth_pct": number } ],
  "profit_loss": [ { "fiscal_year": "FY24", "revenue": number, "sales": number, "ebitda": number, "ebitda_margin_pct": number, "depreciation": number, "ebit": number, "interest": number, "other_income": number, "pbt": number, "tax": number, "tax_rate": number, "reported_pat": number, "adj_pat": number, "eps": number, "adj_eps": number, "shares_outstanding": number, "revenue_growth_pct": number, "ebitda_growth_pct": number, "pat_growth_pct": number } ],
  "balance_sheet": [ { "fiscal_year": "FY24", "cash": number, "accounts_receivable": number, "inventories": number, "other_current_assets": number, "investments": number, "gross_fixed_assets": number, "net_fixed_assets": number, "cwip": number, "intangible_assets": number, "total_assets": number, "accounts_payable": number, "short_term_debt": number, "long_term_debt": number, "total_liabilities": number, "share_capital": number, "reserves": number, "total_equity": number, "working_capital": number, "net_debt": number } ],
  "change_in_estimate": [ { "fiscal_year": "FY25E", "old_revenue": number, "new_revenue": number, "old_ebitda": number, "new_ebitda": number, "old_ebitda_margin_pct": number, "new_ebitda_margin_pct": number, "old_adj_pat": number, "new_adj_pat": number, "old_eps": number, "new_eps": number, "revenue_change_pct": number, "ebitda_change_pct": number, "pat_change_pct": number, "eps_change_pct": number } ],
  "ratios": [ { "fiscal_year": "FY24", "roe": number, "roa": number, "roce": number, "current_ratio": number, "quick_ratio": number, "debt_to_equity": number, "interest_coverage": number, "pe_ratio": number, "pb_ratio": number } ],
  "cash_flow": [ { "fiscal_year": "FY24", "operating_cash_flow": number, "investing_cash_flow": number, "capex": number, "financing_cash_flow": number, "free_cash_flow": number } ],
  "shareholding_percentage": [ { "as_of_date": "YYYY-MM-DD", "promoter_holding_pct": number, "public_holding_pct": number, "institutional_holding_pct": number, "fii_holding_pct": number, "dii_holding_pct": number, "retail_holding_pct": number, "pledged_shares_pct": number } ],
  "price_performance": [ { "date": "YYYY-MM-DD", "open_price": number, "high_price": number, "low_price": number, "close_price": number, "volume": number, "return_1d_pct": number, "return_1w_pct": number, "return_1m_pct": number, "return_3m_pct": number, "return_6m_pct": number, "return_1y_pct": number, "return_3y_pct": number, "return_5y_pct": number, "high_52w": number, "low_52w": number } ]
}`;

  // Build input object for zerox
  const request = { prompt };
  if (inputText && inputText.length > 0) {
    request.text = inputText;
  } else if (pdfPath) {
    request.pdf = pdfPath;
  } else {
    console.error(JSON.stringify({ error: 'NO_INPUT' }));
    process.exit(1);
  }

  try {
    // The actual API varies by zerox version; we attempt a generic call signature.
    // Many zerox demos expose a `run` or `extract` method that accepts an object with
    // a prompt and an input (text or file). We'll try a permissive dynamic call.
    if (!runZerox || typeof runZerox !== 'function') {
      throw new Error('zerox module did not expose a function');
    }

    // Define a JSON schema for structured extraction
    const schema = {
      type: 'object',
      properties: {
        company_name: { type: 'string' },
        sector: { type: ['string', 'null'] },
        industry: { type: ['string', 'null'] },
        profit_loss_quarterly: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              period: { type: 'string' },
              quarter: { type: ['number', 'null'] },
              fiscal_year: { type: ['number', 'string', 'null'] },
              sales: { type: ['number', 'null'] },
              revenue: { type: ['number', 'null'] },
              ebitda: { type: ['number', 'null'] },
              ebitda_margin_pct: { type: ['number', 'null'] },
              depreciation: { type: ['number', 'null'] },
              ebit: { type: ['number', 'null'] },
              interest: { type: ['number', 'null'] },
              other_income: { type: ['number', 'null'] },
              exceptional_items: { type: ['number', 'null'] },
              pbt: { type: ['number', 'null'] },
              tax: { type: ['number', 'null'] },
              reported_pat: { type: ['number', 'null'] },
              adj_pat: { type: ['number', 'null'] },
              eps: { type: ['number', 'null'] },
              adj_eps: { type: ['number', 'null'] },
              shares_outstanding: { type: ['number', 'null'] },
              revenue_yoy_growth_pct: { type: ['number', 'null'] },
              revenue_qoq_growth_pct: { type: ['number', 'null'] },
              ebitda_yoy_growth_pct: { type: ['number', 'null'] },
              ebitda_qoq_growth_pct: { type: ['number', 'null'] },
            },
          },
        },
        profit_loss: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              fiscal_year: { type: ['string', 'number'] },
              revenue: { type: ['number', 'null'] },
              sales: { type: ['number', 'null'] },
              ebitda: { type: ['number', 'null'] },
              ebitda_margin_pct: { type: ['number', 'null'] },
              depreciation: { type: ['number', 'null'] },
              ebit: { type: ['number', 'null'] },
              interest: { type: ['number', 'null'] },
              other_income: { type: ['number', 'null'] },
              pbt: { type: ['number', 'null'] },
              tax: { type: ['number', 'null'] },
              tax_rate: { type: ['number', 'null'] },
              reported_pat: { type: ['number', 'null'] },
              adj_pat: { type: ['number', 'null'] },
              eps: { type: ['number', 'null'] },
              adj_eps: { type: ['number', 'null'] },
              shares_outstanding: { type: ['number', 'null'] },
              revenue_growth_pct: { type: ['number', 'null'] },
              ebitda_growth_pct: { type: ['number', 'null'] },
              pat_growth_pct: { type: ['number', 'null'] },
            },
          },
        },
        balance_sheet: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              fiscal_year: { type: ['string', 'number'] },
              cash: { type: ['number', 'null'] },
              accounts_receivable: { type: ['number', 'null'] },
              inventories: { type: ['number', 'null'] },
              other_current_assets: { type: ['number', 'null'] },
              investments: { type: ['number', 'null'] },
              gross_fixed_assets: { type: ['number', 'null'] },
              net_fixed_assets: { type: ['number', 'null'] },
              cwip: { type: ['number', 'null'] },
              intangible_assets: { type: ['number', 'null'] },
              total_assets: { type: ['number', 'null'] },
              accounts_payable: { type: ['number', 'null'] },
              short_term_debt: { type: ['number', 'null'] },
              long_term_debt: { type: ['number', 'null'] },
              total_liabilities: { type: ['number', 'null'] },
              share_capital: { type: ['number', 'null'] },
              reserves: { type: ['number', 'null'] },
              total_equity: { type: ['number', 'null'] },
              working_capital: { type: ['number', 'null'] },
              net_debt: { type: ['number', 'null'] },
            },
          },
        },
        change_in_estimate: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              fiscal_year: { type: ['string', 'number'] },
              old_revenue: { type: ['number', 'null'] },
              new_revenue: { type: ['number', 'null'] },
              old_ebitda: { type: ['number', 'null'] },
              new_ebitda: { type: ['number', 'null'] },
              old_ebitda_margin_pct: { type: ['number', 'null'] },
              new_ebitda_margin_pct: { type: ['number', 'null'] },
              old_adj_pat: { type: ['number', 'null'] },
              new_adj_pat: { type: ['number', 'null'] },
              old_eps: { type: ['number', 'null'] },
              new_eps: { type: ['number', 'null'] },
              revenue_change_pct: { type: ['number', 'null'] },
              ebitda_change_pct: { type: ['number', 'null'] },
              pat_change_pct: { type: ['number', 'null'] },
              eps_change_pct: { type: ['number', 'null'] },
            },
          },
        },
        ratios: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              fiscal_year: { type: ['string', 'number'] },
              roe: { type: ['number', 'null'] },
              roa: { type: ['number', 'null'] },
              roce: { type: ['number', 'null'] },
              current_ratio: { type: ['number', 'null'] },
              quick_ratio: { type: ['number', 'null'] },
              debt_to_equity: { type: ['number', 'null'] },
              interest_coverage: { type: ['number', 'null'] },
              pe_ratio: { type: ['number', 'null'] },
              pb_ratio: { type: ['number', 'null'] },
            },
          },
        },
        cash_flow: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              fiscal_year: { type: ['string', 'number'] },
              operating_cash_flow: { type: ['number', 'null'] },
              investing_cash_flow: { type: ['number', 'null'] },
              capex: { type: ['number', 'null'] },
              financing_cash_flow: { type: ['number', 'null'] },
              free_cash_flow: { type: ['number', 'null'] },
            },
          },
        },
        shareholding_percentage: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              as_of_date: { type: ['string', 'null'] },
              promoter_holding_pct: { type: ['number', 'null'] },
              public_holding_pct: { type: ['number', 'null'] },
              institutional_holding_pct: { type: ['number', 'null'] },
              fii_holding_pct: { type: ['number', 'null'] },
              dii_holding_pct: { type: ['number', 'null'] },
              retail_holding_pct: { type: ['number', 'null'] },
              pledged_shares_pct: { type: ['number', 'null'] },
            },
          },
        },
        price_performance: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              date: { type: ['string', 'null'] },
              open_price: { type: ['number', 'null'] },
              high_price: { type: ['number', 'null'] },
              low_price: { type: ['number', 'null'] },
              close_price: { type: ['number', 'null'] },
              volume: { type: ['number', 'null'] },
              return_1d_pct: { type: ['number', 'null'] },
              return_1w_pct: { type: ['number', 'null'] },
              return_1m_pct: { type: ['number', 'null'] },
              return_3m_pct: { type: ['number', 'null'] },
              return_6m_pct: { type: ['number', 'null'] },
              return_1y_pct: { type: ['number', 'null'] },
              return_3y_pct: { type: ['number', 'null'] },
              return_5y_pct: { type: ['number', 'null'] },
              high_52w: { type: ['number', 'null'] },
              low_52w: { type: ['number', 'null'] },
            },
          },
        },
      },
    };

    const openaiAPIKey = process.env.OPENAI_API_KEY || '';
    const result = await runZerox({
      filePath: pdfPath,
      prompt,
      schema,
      openaiAPIKey,
      // Keep defaults for model/provider (OpenAI GPT-4o-mini)
    });

    const out = result && (result.extracted || result);
    console.log(JSON.stringify(out));
  } catch (err) {
    console.error(JSON.stringify({ error: 'ZER0X_RUN_FAILED', message: String(err) }));
    process.exit(3);
  }
}

main().catch((e) => {
  console.error(JSON.stringify({ error: 'ZER0X_MAIN_FAILED', message: String(e) }));
  process.exit(4);
});
"""


def _write_zerox_runner(tmpdir: Path) -> Path:
    """Write the transient zerox runner script to tmpdir and return its path."""
    runner = tmpdir / "index.mjs"
    runner.write_text(ZER0X_RUNNER_JS, encoding="utf-8")
    os.chmod(runner, 0o755)
    return runner


def _ensure_node_and_zerox(tmpdir: Path) -> bool:
    """Best-effort check to ensure Node is present and zerox can be imported.

    Returns True if we can attempt the zerox runner, False otherwise.
    """
    try:
        subprocess.run(["node", "-v"], check=True, capture_output=True)
    except Exception:
        return False

    # Try to run a quick import test for zerox in a throwaway script
    test_js = tmpdir / "test_zerox_import.mjs"
    test_js.write_text(
        """
import { createRequire } from 'module';
const base = process.env.ZEROX_BASE || process.cwd();
const requireFromBase = createRequire(base.endsWith('/') ? base : base + '/');
try {
  requireFromBase('zerox');
  console.log('OK');
} catch (e) {
  console.log('NO');
  process.exit(2);
}
        """,
        encoding="utf-8",
    )
    try:
        r = subprocess.run(["node", str(test_js)], capture_output=True, text=True)
        ok = (r.returncode == 0 and "OK" in (r.stdout or r.stderr))
        return ok
    except Exception:
        return False


def _extract_text_from_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """Extract text content from PDF, page by page (fallback path)."""
    if not _pypdf2_available:
        raise RuntimeError("PyPDF2 not available for text extraction")

    pages: list[dict[str, Any]] = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    'page_number': page_num + 1,
                    'text': text,
                })
    return pages


def _parse_with_gpt5(pages: list[dict[str, Any]], company_hint: str | None = None) -> dict[str, Any]:
    """Use GPT-5 (via ai-sdk) to parse financial data from PDF text.

    Includes sections: profit_loss_quarterly, profit_loss, balance_sheet,
    change_in_estimate, ratios, cash_flow, shareholding_percentage, price_performance.
    """
    if not _ai_sdk_available:
        raise RuntimeError("ai_sdk not available for GPT parsing")

    s = load_settings()
    full_text = "\n\n---PAGE BREAK---\n\n".join([f"[Page {p['page_number']}]\n{p['text']}" for p in pages])

    system_prompt = f"""You are a financial data extraction expert. Extract structured financial data from the provided PDF text.

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
1) Extract ALL available data; use null if missing.
2) Convert percentage strings to numbers in percent units (e.g., 12.4% -> 12.4).
3) Remove thousands separators from numbers (e.g., 2,678 -> 2678).
4) Parse multiple fiscal years and quarters where present.
5) If a section is not present, return an empty array for that section.
6) Return ONLY valid JSON. No markdown fences or commentary.
"""

    model = openai(s.openai_model)
    messages = [
        CoreSystemMessage(content=system_prompt),
        CoreUserMessage(content=[TextPart(text=f"PDF Text Content:\n\n{full_text}")])
    ]

    resp = generate_text(model=model, messages=messages)
    text = resp.text.strip()
    # strip markdown code fences if present
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\n', '', text)
        text = re.sub(r'\n```$', '', text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {resp.text}")

    return data


def _extract_with_zerox_or_text(pdf_path: Path, company_hint: str | None) -> dict[str, Any]:
    """Try zerox runner first (if available), then fall back to GPT-5 text parsing."""
    # Attempt zerox in a temp dir
    with tempfile.TemporaryDirectory(prefix="zerox_runner_") as td:
        tmpdir = Path(td)
        runner = _write_zerox_runner(tmpdir)

        if _ensure_node_and_zerox(tmpdir):
            try:
                # Pass PDF path directly to runner
                env = os.environ.copy()
                # Honor OPENAI_API_KEY if present
                from .config import load_settings as _load
                s = _load()
                if s.openai_api_key:
                    env["OPENAI_API_KEY"] = s.openai_api_key
                # Model hint for zerox (if it respects it)
                env.setdefault("OPENAI_MODEL", s.openai_model)
                # Help the Node runner resolve from repo root if zerox is installed there
                env.setdefault("ZEROX_BASE", str(Path.cwd()))

                # Convert to absolute path to avoid path resolution issues in zerox
                abs_pdf_path = pdf_path.resolve() if not pdf_path.is_absolute() else pdf_path

                proc = subprocess.run(
                    ["node", str(runner), "--pdf", str(abs_pdf_path)],
                    capture_output=True,
                    text=True,
                    env=env,
                    cwd=tmpdir,
                )

                if proc.returncode == 0 and proc.stdout.strip():
                    try:
                        return json.loads(proc.stdout)
                    except Exception:
                        pass
                # If zerox failed, fall through to text
            except Exception:
                pass

    # Fallback: text extraction + GPT-5
    if not _pypdf2_available or not _ai_sdk_available:
        raise RuntimeError("Neither zerox nor text-based parsing is available")

    pages = _extract_text_from_pdf(str(pdf_path))
    data = _parse_with_gpt5(pages, company_hint=company_hint)
    data['parsing_method'] = 'zerox-text-fallback'  # record fallback path
    data['pdf_path'] = str(pdf_path)
    return data


def _extract_with_zerox_only(pdf_path: Path, company_hint: str | None) -> dict[str, Any]:
    """Run extraction strictly via zerox; raise if unavailable or fails."""
    with tempfile.TemporaryDirectory(prefix="zerox_runner_") as td:
        tmpdir = Path(td)
        runner = _write_zerox_runner(tmpdir)

        if not _ensure_node_and_zerox(tmpdir):
            raise RuntimeError("zerox module not available; run `npm i zerox` at repo root")

        env = os.environ.copy()
        s = load_settings()
        if s.openai_api_key:
            env["OPENAI_API_KEY"] = s.openai_api_key
        env.setdefault("OPENAI_MODEL", s.openai_model)
        env.setdefault("ZEROX_BASE", str(Path.cwd()))

        # Convert to absolute path to avoid path resolution issues in zerox
        abs_pdf_path = pdf_path.resolve() if not pdf_path.is_absolute() else pdf_path

        proc = subprocess.run(
            ["node", str(runner), "--pdf", str(abs_pdf_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmpdir,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"zerox runner failed: {proc.stderr.strip() or proc.stdout.strip()}")
        try:
            return json.loads(proc.stdout)
        except Exception as e:
            raise RuntimeError(f"zerox output not JSON: {e}\nOutput: {proc.stdout[:4000]}")


# --------------------------------------------------------------------------------------
# Insert helpers for additional tables not covered by geojit.ingest helpers
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

def ingest_zerox(max_files: int | None = None, *, file: str | None = None, zerox_only: bool = False, no_db: bool = False) -> None:
    """Ingest all PDFs using zerox (if available) or GPT-5 text fallback.

    - Discovers PDFs under Settings.data_dir
    - Ensures DB schema
    - For each PDF: upsert document, company and organization, then insert all tables
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

            company_hint = path.stem.split('_')[0] if '_' in path.stem else path.stem
            if zerox_only:
                parsed = _extract_with_zerox_only(path, company_hint=company_hint)
            else:
                parsed = _extract_with_zerox_or_text(path, company_hint=company_hint)

            company_name = parsed.get('company_name') or company_hint
            sector = parsed.get('sector')
            industry = parsed.get('industry')

            org_id = None
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
                    parsing_method=parsed.get('parsing_method', 'zerox'),
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

            # Build embeddings and upsert to Qdrant (no Postgres chunks)
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

            # Upsert a company-name vector into a dedicated companies collection for fuzzy lookup
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

            # Optionally dump parsed JSON next to PDF for evaluation
            try:
                out_json = Path.cwd() / f"parsed_output_{path.stem}.json"
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
            errors += 1

    print(f"Done. Success={success}, Errors={errors}")


def eval_zerox(pdf_path: str, csv_paths: list[str]) -> dict[str, Any]:
    """Run zerox-only extraction for a single PDF and evaluate against CSVs."""
    p = Path(pdf_path)
    company_hint = p.stem
    parsed = _extract_with_zerox_only(p, company_hint=company_hint)

    # Save for inspection
    out_json = Path.cwd() / f"parsed_output_{p.stem}.json"
    with open(out_json, 'w') as f:
        json.dump(parsed, f, indent=2, default=str)

    results = evaluate_parser_output(parsed, csv_paths)
    print_evaluation_report(results)
    # Also save raw results
    eval_out = Path.cwd() / f"eval_results_{p.stem}.json"
    with open(eval_out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="ingest_zerox", description="Ingest PDFs using zerox (with GPT fallback)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of PDFs to process")
    parser.add_argument("--file", type=str, default=None, help="Path to a single PDF to process")
    parser.add_argument("--zerox-only", action="store_true", help="Require zerox; do not fallback to text-based parsing")
    parser.add_argument("--no-db", action="store_true", help="Do not write to DB; only parse and write JSON")
    parser.add_argument("--eval", action="store_true", help="Run evaluation against CSV ground truth for a single PDF")
    parser.add_argument("--eval-csv", action="append", default=None, help="Path to a CSV file (repeatable)")
    args = parser.parse_args()

    if args.eval:
        if not args.file:
            raise SystemExit("--eval requires --file to be specified")
        if not args.eval_csv:
            raise SystemExit("--eval requires at least one --eval-csv path")
        eval_zerox(args.file, args.eval_csv)
    else:
        ingest_zerox(max_files=args.max_files, file=args.file, zerox_only=args.zerox_only, no_db=args.no_db)
