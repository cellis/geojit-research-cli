#!/usr/bin/env python3
"""Test script for text-based PDF parser with evaluation against CSV ground truth."""

import sys
from pathlib import Path

from geojit.pdf_parser_text import parse_pdf_text_based
from geojit.pdf_evaluator import evaluate_parser_output, print_evaluation_report


def main():
    # PDF file to test (TTK - the one we have CSV evals for)
    pdf_path = "/Users/cameronellis/work/geojit-research-cli/Financial_Research_Agent_Files/SP20241406115209223TTK.pdf"

    # CSV ground truth files
    csv_files = [
        "/Users/cameronellis/work/geojit-research-cli/SP20241406115209223TTK - quarterly_profit_loss.csv",
        "/Users/cameronellis/work/geojit-research-cli/SP20241406115209223TTK - profit_loss.csv",
        "/Users/cameronellis/work/geojit-research-cli/SP20241406115209223TTK - balance_sheet.csv",
        "/Users/cameronellis/work/geojit-research-cli/SP20241406115209223TTK - change_in_estimates.csv",
    ]

    # Verify files exist
    if not Path(pdf_path).exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    for csv_file in csv_files:
        if not Path(csv_file).exists():
            print(f"Warning: CSV not found: {csv_file}")

    print("="*80)
    print("TEXT-BASED PDF PARSER TEST")
    print("="*80)
    print(f"\nPDF: {Path(pdf_path).name}")
    print(f"CSV Evals: {len([f for f in csv_files if Path(f).exists()])}")
    print()

    # Parse PDF
    try:
        parsed_data = parse_pdf_text_based(pdf_path, company_hint="TTK")
        print("\n[SUCCESS] PDF parsed successfully")

        # Save parsed output for inspection
        import json
        output_file = "/Users/cameronellis/work/geojit-research-cli/parsed_output_text.json"
        with open(output_file, 'w') as f:
            json.dump(parsed_data, f, indent=2)
        print(f"[INFO] Parsed data saved to: {output_file}")

    except Exception as e:
        print(f"\n[ERROR] Failed to parse PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Evaluate against CSV
    print("\n[INFO] Evaluating against ground truth CSV files...")
    existing_csvs = [f for f in csv_files if Path(f).exists()]

    if existing_csvs:
        results = evaluate_parser_output(parsed_data, existing_csvs)
        print_evaluation_report(results)

        # Save evaluation results
        import json
        eval_output = "/Users/cameronellis/work/geojit-research-cli/eval_results_text.json"
        with open(eval_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] Evaluation results saved to: {eval_output}")
    else:
        print("[WARNING] No CSV files found for evaluation")


if __name__ == "__main__":
    main()
