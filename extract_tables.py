#!/usr/bin/env python3
"""
Financial table extraction agent using GPT-5 vision.
Extracts specific tables from research PDFs and validates against ground truth CSVs.
"""

import base64
import json
import sys
from pathlib import Path
import fitz  # PyMuPDF
from openai import OpenAI
import pandas as pd


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[bytes]:
    """Convert PDF pages to PNG images."""
    images = []
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        images.append(png_bytes)

    doc.close()
    return images


def encode_image(image_bytes: bytes) -> str:
    """Encode image bytes to base64."""
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_tables_with_vision(images: list[bytes], company_hint: str, api_key: str) -> dict:
    """Use GPT-5 vision to extract financial tables from PDF images."""

    client = OpenAI(api_key=api_key)

    system_prompt = """You are a financial data extraction expert. Extract ALL financial tables from the provided PDF pages.

Return ONLY a valid JSON object with this EXACT structure (use null for missing values):

{
  "company_name": "string",
  "sector": "string|null",
  "industry": "string|null",

  "profit_loss": [
    {
      "fiscal_year": "FY24",
      "revenue": number,
      "operating_expenses": number,
      "ebitda": number,
      "ebitda_margin_pct": number,
      "depreciation": number,
      "ebit": number,
      "interest": number,
      "other_income": number,
      "pbt": number,
      "tax": number,
      "pat": number,
      "eps": number,
      "shares_outstanding": number
    }
  ],

  "profit_loss_quarterly": [
    {
      "period": "Q4FY24",
      "quarter": 4,
      "fiscal_year": 2024,
      "revenue": number,
      "operating_expenses": number,
      "ebitda": number,
      "ebitda_margin_pct": number,
      "depreciation": number,
      "ebit": number,
      "interest": number,
      "other_income": number,
      "pbt": number,
      "tax": number,
      "pat": number,
      "eps": number,
      "revenue_yoy_growth_pct": number
    }
  ],

  "balance_sheet": [
    {
      "fiscal_year": "FY24",
      "cash": number,
      "accounts_receivable": number,
      "inventories": number,
      "current_assets": number,
      "gross_fixed_assets": number,
      "net_fixed_assets": number,
      "investments": number,
      "total_assets": number,
      "accounts_payable": number,
      "short_term_debt": number,
      "current_liabilities": number,
      "long_term_debt": number,
      "total_liabilities": number,
      "share_capital": number,
      "reserves": number,
      "total_equity": number,
      "net_debt": number
    }
  ],

  "cash_flow": [
    {
      "fiscal_year": "FY24",
      "operating_cash_flow": number,
      "investing_cash_flow": number,
      "financing_cash_flow": number,
      "capex": number,
      "free_cash_flow": number,
      "net_change_in_cash": number
    }
  ],

  "ratios": [
    {
      "fiscal_year": "FY24",
      "roe": number,
      "roa": number,
      "roce": number,
      "current_ratio": number,
      "debt_to_equity": number,
      "interest_coverage": number,
      "pe_ratio": number,
      "pb_ratio": number,
      "dividend_yield": number
    }
  ],

  "change_in_estimate": [
    {
      "fiscal_year": "FY25E",
      "old_revenue": number,
      "new_revenue": number,
      "old_ebitda": number,
      "new_ebitda": number,
      "old_pat": number,
      "new_pat": number,
      "old_eps": number,
      "new_eps": number
    }
  ]
}

CRITICAL INSTRUCTIONS:
1. Extract ALL fiscal years present in each table
2. Convert percentages to numeric values (e.g., "12.5%" -> 12.5)
3. Remove commas from numbers (e.g., "2,678" -> 2678)
4. Preserve negative numbers with minus sign
5. Use null if a value is missing or not present
6. Return ONLY valid JSON - no markdown fences, no commentary
7. Analyze ALL pages - tables often span multiple pages
8. Match column headers precisely to identify fields correctly
9. For quarterly data, extract period (e.g., "Q4FY24"), quarter number, and fiscal year
"""

    # Build message with all images
    content = [{"type": "text", "text": f"Extract all financial tables from this PDF for company: {company_hint}"}]

    for img_bytes in images:
        base64_img = encode_image(img_bytes)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_img}",
                "detail": "high"
            }
        })

    content.append({
        "type": "text",
        "text": "Now extract ALL financial tables as structured JSON following the schema above."
    })

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content}
            ],
            max_completion_tokens=16000,
        )

        text = response.choices[0].message.content.strip()

        # Strip markdown fences
        if text.startswith('```'):
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines)

        data = json.loads(text)
        return data

    except Exception as e:
        raise RuntimeError(f"GPT-5 vision extraction failed: {e}")


def evaluate_extraction(parsed_data: dict, csv_paths: list[str]) -> dict:
    """Compare extracted data against ground truth CSVs."""

    results = {
        "overall_accuracy": 0,
        "table_results": {}
    }

    for csv_path in csv_paths:
        if not Path(csv_path).exists():
            print(f"Warning: CSV not found: {csv_path}")
            continue

        # Identify table type from filename
        csv_name = Path(csv_path).name.lower()
        if 'quarterly_profit_loss' in csv_name:
            table_key = 'profit_loss_quarterly'
        elif 'profit_loss' in csv_name:
            table_key = 'profit_loss'
        elif 'balance_sheet' in csv_name:
            table_key = 'balance_sheet'
        elif 'cash_flow' in csv_name:
            table_key = 'cash_flow'
        elif 'ratios' in csv_name:
            table_key = 'ratios'
        elif 'change_in_estimate' in csv_name:
            table_key = 'change_in_estimate'
        else:
            continue

        # Load ground truth
        df_truth = pd.read_csv(csv_path)
        extracted = parsed_data.get(table_key, [])

        # Basic metrics
        truth_rows = len(df_truth)
        extracted_rows = len(extracted) if isinstance(extracted, list) else 0

        # Calculate field coverage
        if truth_rows > 0 and extracted_rows > 0:
            truth_cols = set(df_truth.columns)
            extracted_cols = set(extracted[0].keys()) if extracted else set()
            matching_cols = truth_cols.intersection(extracted_cols)
            coverage = len(matching_cols) / len(truth_cols) * 100
        else:
            coverage = 0

        results["table_results"][table_key] = {
            "csv_file": csv_path,
            "truth_rows": truth_rows,
            "extracted_rows": extracted_rows,
            "field_coverage_pct": round(coverage, 1),
            "status": "✓" if coverage > 70 else "✗"
        }

    # Calculate overall accuracy
    if results["table_results"]:
        avg_coverage = sum(r["field_coverage_pct"] for r in results["table_results"].values()) / len(results["table_results"])
        results["overall_accuracy"] = round(avg_coverage, 1)

    return results


def print_evaluation(results: dict):
    """Print evaluation results."""
    print("\n" + "="*60)
    print("EXTRACTION EVALUATION RESULTS")
    print("="*60)

    for table, metrics in results["table_results"].items():
        print(f"\n{metrics['status']} {table.upper()}")
        print(f"  Ground truth rows: {metrics['truth_rows']}")
        print(f"  Extracted rows: {metrics['extracted_rows']}")
        print(f"  Field coverage: {metrics['field_coverage_pct']}%")

    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {results['overall_accuracy']}%")
    print("="*60 + "\n")


def main():
    """Main extraction workflow."""

    # Get API key from environment
    import os
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Test on TTK PDF first
    pdf_path = Path("Financial_Research_Agent_Files/SP20241406115209223TTK.pdf")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Processing: {pdf_path.name}")
    print("Converting PDF to images...")

    images = pdf_to_images(pdf_path, dpi=200)
    print(f"Converted {len(images)} pages")

    print("Extracting tables with GPT-5 vision...")
    company_hint = pdf_path.stem.split('_')[0] if '_' in pdf_path.stem else pdf_path.stem

    parsed = extract_tables_with_vision(images, company_hint, api_key)

    # Save output
    output_file = Path(f"parsed_output_{pdf_path.stem}.json")
    with open(output_file, 'w') as f:
        json.dump(parsed, f, indent=2)
    print(f"Saved: {output_file}")

    # Evaluate against ground truth
    csv_paths = [
        "SP20241406115209223TTK - quarterly_profit_loss.csv",
        "SP20241406115209223TTK - balance_sheet.csv",
        "SP20241406115209223TTK - change_in_estimates.csv",
        "SP20241406115209223TTK - profit_loss.csv"
    ]

    print("\nEvaluating extraction accuracy...")
    results = evaluate_extraction(parsed, csv_paths)
    print_evaluation(results)

    # Save evaluation results
    eval_file = Path(f"eval_results_{pdf_path.stem}.json")
    with open(eval_file, 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
