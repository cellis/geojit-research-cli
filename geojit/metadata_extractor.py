from __future__ import annotations

import re
from typing import Any

try:
    from ai_sdk import generate_object, openai
    from pydantic import BaseModel, Field
    _ai_sdk_available = True
except Exception:
    _ai_sdk_available = False


class CompanyMetadata(BaseModel):
    """Structured company metadata extracted from financial documents."""
    company_name: str = Field(description="Full company name")
    sector: str | None = Field(default=None, description="Business sector (e.g., Pharma, Banking, Auto)")
    industry: str | None = Field(default=None, description="Specific industry")
    report_period: str | None = Field(default=None, description="Report period (e.g., Q4FY24, FY2023)")
    fiscal_year: int | None = Field(default=None, description="Fiscal year as integer (e.g., 2024)")
    quarter: int | None = Field(default=None, description="Quarter number if applicable (1-4)")


class FinancialMetric(BaseModel):
    """A single financial metric extracted from a document."""
    metric_name: str = Field(description="Name of the metric (e.g., revenue, profit, ebitda_margin)")
    value: float | None = Field(default=None, description="Numeric value")
    value_text: str = Field(description="Original text representation (e.g., 'Rs 798 crore')")
    unit: str | None = Field(default=None, description="Unit (e.g., 'crore', 'million', 'percent')")
    period: str | None = Field(default=None, description="Time period for this metric")


class DocumentMetadata(BaseModel):
    """Complete metadata for a financial document."""
    company: CompanyMetadata
    metrics: list[FinancialMetric] = Field(default_factory=list)


def extract_metadata_with_llm(text: str, api_key: str, model: str = "gpt-4o-mini") -> DocumentMetadata | None:
    """Extract structured metadata from document text using LLM."""
    if not _ai_sdk_available:
        return None

    # Truncate text to first ~4000 characters (usually enough for header + key metrics)
    sample_text = text[:4000]

    prompt = f"""Extract structured metadata from this financial research document.

Document excerpt:
{sample_text}

Extract:
1. Company name (full legal name if possible)
2. Sector and industry
3. Report period (e.g., Q4FY24, FY2023)
4. Key financial metrics with their values

For metrics, extract common ones like:
- Revenue/Sales
- Profit (Net Profit, Operating Profit)
- EBITDA and EBITDA Margin
- Total Assets
- Equity
- Growth rates

Be precise with numbers and units."""

    try:
        llm_model = openai(model)
        result = generate_object(
            model=llm_model,
            schema=DocumentMetadata,
            prompt=prompt
        )
        return result.object
    except Exception as e:
        print(f"LLM metadata extraction failed: {e}")
        return None


def extract_metadata_heuristic(text: str, filename: str) -> dict[str, Any]:
    """Extract metadata using heuristic patterns (fallback method)."""
    # Try to extract company name from filename
    # Common pattern: SP20241006120459650BATA.pdf -> BATA
    # Or: "Sun PharmaQ4.pdf" -> Sun Pharma
    company_name = None

    # Remove common prefixes and date patterns
    clean_filename = re.sub(r'^SP\d+', '', filename)
    clean_filename = re.sub(r'\d{8,}', '', clean_filename)
    clean_filename = re.sub(r'\.pdf$', '', clean_filename, flags=re.IGNORECASE)
    clean_filename = clean_filename.strip('_- ')

    # Try to extract period info (Q1, Q2, Q3, Q4, FY)
    period_match = re.search(r'(Q[1-4]|FY)(\d{2,4})?', text, re.IGNORECASE)
    period = period_match.group(0) if period_match else None

    # Extract fiscal year
    fy_match = re.search(r'FY\s*(\d{2,4})', text, re.IGNORECASE)
    fiscal_year = None
    if fy_match:
        year_str = fy_match.group(1)
        fiscal_year = int(year_str) if len(year_str) == 4 else (2000 + int(year_str))

    # Look for company name in first few lines
    lines = text.split('\n')[:10]
    for line in lines:
        line = line.strip()
        # Company names are often in title case and at the start
        if len(line) > 5 and len(line) < 100 and line[0].isupper():
            # Skip common report headers
            if any(skip in line.lower() for skip in ['result', 'update', 'quarter', 'report', 'research']):
                continue
            company_name = line
            break

    if not company_name and clean_filename:
        company_name = clean_filename

    return {
        "company_name": company_name,
        "period": period,
        "fiscal_year": fiscal_year,
        "source": "heuristic"
    }


def extract_document_metadata(text: str, filename: str, api_key: str | None, use_llm: bool = True) -> dict[str, Any]:
    """
    Extract metadata from a financial document.
    Falls back to heuristics if LLM fails or is disabled.
    """
    # Try LLM extraction first if enabled and available
    if use_llm and api_key and _ai_sdk_available:
        try:
            metadata = extract_metadata_with_llm(text, api_key)
            if metadata:
                return {
                    "company_name": metadata.company.company_name,
                    "sector": metadata.company.sector,
                    "industry": metadata.company.industry,
                    "period": metadata.company.report_period,
                    "fiscal_year": metadata.company.fiscal_year,
                    "quarter": metadata.company.quarter,
                    "metrics": [
                        {
                            "name": m.metric_name,
                            "value": m.value,
                            "value_text": m.value_text,
                            "unit": m.unit,
                            "period": m.period
                        }
                        for m in metadata.metrics
                    ],
                    "source": "llm"
                }
        except Exception as e:
            print(f"LLM extraction failed, falling back to heuristics: {e}")

    # Fallback to heuristic extraction
    return extract_metadata_heuristic(text, filename)
