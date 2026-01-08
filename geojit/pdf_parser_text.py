"""Text-based PDF parser using GPT-5 for structured financial data extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from ai_sdk import generate_text, openai
    from ai_sdk.types import CoreSystemMessage, CoreUserMessage, TextPart
    _ai_sdk_available = True
except Exception:
    _ai_sdk_available = False

try:
    import PyPDF2
    _pypdf2_available = True
except Exception:
    _pypdf2_available = False

_dependencies_available = _ai_sdk_available and _pypdf2_available

from .config import load_settings


def extract_text_from_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """Extract text content from PDF, page by page."""
    if not _dependencies_available:
        raise RuntimeError("Required dependencies not available (PyPDF2, ai_sdk)")

    pages = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    'page_number': page_num + 1,
                    'text': text
                })

    return pages


def parse_financial_data_with_llm(pages: list[dict[str, Any]], company_hint: str | None = None) -> dict[str, Any]:
    """
    Use GPT-5 to parse financial data from PDF text.
    Returns structured data matching our database schema.
    """
    if not _dependencies_available:
        raise RuntimeError("ai_sdk not available")

    s = load_settings()

    # Combine all pages with markers
    full_text = "\n\n---PAGE BREAK---\n\n".join([
        f"[Page {p['page_number']}]\n{p['text']}" for p in pages
    ])

    system_prompt = f"""You are a financial data extraction expert. Extract structured financial data from the provided PDF text.

Company hint: {company_hint or 'Unknown'}

Extract the following information in JSON format:

{{
  "company_name": "string",
  "sector": "string or null",
  "industry": "string or null",

  "profit_loss_quarterly": [
    {{
      "period": "Q4FY24",
      "quarter": 4,
      "fiscal_year": 2024,
      "sales": number,
      "revenue": number,
      "ebitda": number,
      "ebitda_margin_pct": number,
      "depreciation": number,
      "ebit": number,
      "interest": number,
      "other_income": number,
      "exceptional_items": number,
      "pbt": number,
      "tax": number,
      "reported_pat": number,
      "adj_pat": number,
      "eps": number,
      "adj_eps": number,
      "shares_outstanding": number,
      "revenue_yoy_growth_pct": number,
      "revenue_qoq_growth_pct": number,
      "ebitda_yoy_growth_pct": number,
      "ebitda_qoq_growth_pct": number
    }}
  ],

  "profit_loss": [
    {{
      "fiscal_year": "FY24",
      "revenue": number,
      "sales": number,
      "ebitda": number,
      "ebitda_margin_pct": number,
      "depreciation": number,
      "ebit": number,
      "interest": number,
      "other_income": number,
      "pbt": number,
      "tax": number,
      "tax_rate": number,
      "reported_pat": number,
      "adj_pat": number,
      "eps": number,
      "adj_eps": number,
      "shares_outstanding": number,
      "revenue_growth_pct": number,
      "ebitda_growth_pct": number,
      "pat_growth_pct": number
    }}
  ],

  "balance_sheet": [
    {{
      "fiscal_year": "FY24",
      "cash": number,
      "accounts_receivable": number,
      "inventories": number,
      "other_current_assets": number,
      "investments": number,
      "gross_fixed_assets": number,
      "net_fixed_assets": number,
      "cwip": number,
      "intangible_assets": number,
      "total_assets": number,
      "accounts_payable": number,
      "short_term_debt": number,
      "long_term_debt": number,
      "total_liabilities": number,
      "share_capital": number,
      "reserves": number,
      "total_equity": number,
      "working_capital": number,
      "net_debt": number
    }}
  ],

  "change_in_estimate": [
    {{
      "fiscal_year": "FY25E",
      "old_revenue": number,
      "new_revenue": number,
      "old_ebitda": number,
      "new_ebitda": number,
      "old_ebitda_margin_pct": number,
      "new_ebitda_margin_pct": number,
      "old_adj_pat": number,
      "new_adj_pat": number,
      "old_eps": number,
      "new_eps": number,
      "revenue_change_pct": number,
      "ebitda_change_pct": number,
      "pat_change_pct": number,
      "eps_change_pct": number
    }}
  ],

  "ratios": [
    {{
      "fiscal_year": "FY24",
      "roe": number,
      "roa": number,
      "roce": number,
      "current_ratio": number,
      "quick_ratio": number,
      "debt_to_equity": number,
      "interest_coverage": number,
      "pe_ratio": number,
      "pb_ratio": number
    }}
  ],

  "cash_flow": [
    {{
      "fiscal_year": "FY24",
      "operating_cash_flow": number,
      "investing_cash_flow": number,
      "capex": number,
      "financing_cash_flow": number,
      "free_cash_flow": number
    }}
  ]
}}

IMPORTANT INSTRUCTIONS:
1. Extract ALL available data - don't skip fields even if they seem redundant
2. Use null for missing values
3. Convert percentages to decimals (e.g., 12.4% -> 12.4)
4. Remove commas from numbers (e.g., "2,678" -> 2678)
5. Parse growth rates carefully (e.g., "-3.6%" -> -3.6)
6. Match column names to our schema (sales/revenue/turnover are all valid)
7. Extract data for ALL available fiscal years and quarters
8. If you see "bps" (basis points), convert to percentage points (e.g., "-70bps" -> -0.7)
9. Return ONLY valid JSON, no commentary

If data is clearly not available for a section, return an empty array [] for that section."""

    model = openai(s.openai_model)

    messages = [
        CoreSystemMessage(content=system_prompt),
        CoreUserMessage(content=[TextPart(text=f"PDF Text Content:\n\n{full_text}")])
    ]

    resp = generate_text(model=model, messages=messages)

    # Parse JSON response
    try:
        # Extract JSON from response (handle markdown code blocks)
        text = resp.text.strip()
        if text.startswith('```'):
            # Remove markdown code fences
            text = re.sub(r'^```(?:json)?\n', '', text)
            text = re.sub(r'\n```$', '', text)

        data = json.loads(text)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response as JSON: {e}\nResponse: {resp.text}")


def parse_pdf_text_based(pdf_path: str, company_hint: str | None = None) -> dict[str, Any]:
    """
    Main entry point for text-based PDF parsing.

    Args:
        pdf_path: Path to the PDF file
        company_hint: Optional hint about company name (from filename or metadata)

    Returns:
        Dictionary with structured financial data
    """
    print(f"[TEXT PARSER] Extracting text from {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    print(f"[TEXT PARSER] Extracted {len(pages)} pages")

    print(f"[TEXT PARSER] Parsing with LLM...")
    data = parse_financial_data_with_llm(pages, company_hint)
    print(f"[TEXT PARSER] Parsing complete")

    # Add metadata
    data['parsing_method'] = 'text'
    data['pdf_path'] = str(pdf_path)

    return data
