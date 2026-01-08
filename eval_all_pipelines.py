#!/usr/bin/env python3
"""Evaluate all ingestion pipelines against ground truth CSVs.

Runs each pipeline (ingest.py, ingest_zerox.py, ingest_vision.py) against
the TTK PDF with CSV ground truth, clearing the database between runs.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


# Test file configuration
PDF_FILE = "Financial_Research_Agent_Files/SP20241406115209223TTK.pdf"
CSV_FILES = [
    "SP20241406115209223TTK - balance_sheet.csv",
    "SP20241406115209223TTK - change_in_estimates.csv",
    "SP20241406115209223TTK - profit_loss.csv",
    "SP20241406115209223TTK - quarterly_profit_loss.csv",
]

RESULTS_FILE = "pipeline_eval_results.txt"


def run_command(cmd: list[str], description: str, timeout: int = 600) -> tuple[bool, str]:
    """Run a command and return (success, output)."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        output = result.stdout + result.stderr
        success = result.returncode == 0
        print(output)
        return success, output
    except subprocess.TimeoutExpired:
        msg = f"TIMEOUT after {timeout}s"
        print(msg)
        return False, msg
    except Exception as e:
        msg = f"ERROR: {e}"
        print(msg)
        return False, msg


def clear_database():
    """Clear the geojit-test database."""
    print("\n" + "="*80)
    print("Clearing database...")
    print("="*80 + "\n")

    return run_command(
        ["uv", "run", "python", "recreate_test_db.py"],
        "Clear database",
        timeout=60
    )


def write_results(pipeline_name: str, success: bool, output: str, eval_json_path: str | None = None):
    """Append results to the results file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(RESULTS_FILE, "a") as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Pipeline: {pipeline_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Success: {success}\n")
        f.write("="*80 + "\n\n")

        # Try to extract evaluation metrics from output
        if "Evaluation Results" in output:
            lines = output.split("\n")
            in_eval = False
            for line in lines:
                if "Evaluation Results" in line or in_eval:
                    in_eval = True
                    f.write(line + "\n")
                    if line.strip() == "" and in_eval:
                        break

        # If there's a JSON file, try to extract key metrics
        if eval_json_path and Path(eval_json_path).exists():
            import json
            try:
                with open(eval_json_path) as jf:
                    data = json.load(jf)
                    f.write("\nKey Metrics:\n")
                    for table, metrics in data.items():
                        if isinstance(metrics, dict) and "accuracy" in metrics:
                            f.write(f"  {table}: {metrics['accuracy']:.2%} accuracy\n")
            except Exception:
                pass

        f.write("\nFull Output:\n")
        f.write(output)
        f.write("\n\n")


def main():
    # Clear results file
    with open(RESULTS_FILE, "w") as f:
        f.write(f"Pipeline Evaluation Results\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PDF: {PDF_FILE}\n")
        f.write(f"CSVs: {', '.join(CSV_FILES)}\n")
        f.write("="*80 + "\n")

    pipelines = [
        {
            "name": "ingest.py (text-based)",
            "module": "geojit.ingest",
            "has_eval": True,
            "eval_json_pattern": "eval_results_text_{stem}.json",
        },
        {
            "name": "ingest_zerox.py (zerox/GPT fallback)",
            "module": "geojit.ingest_zerox",
            "has_eval": True,
            "eval_json_pattern": "eval_results_{stem}.json",
        },
        {
            "name": "ingest_vision.py (GPT-5 vision)",
            "module": "geojit.ingest_vision",
            "has_eval": True,
            "eval_json_pattern": "eval_results_vision_{stem}.json",
        },
    ]

    for pipeline in pipelines:
        print(f"\n{'#'*80}")
        print(f"# Testing: {pipeline['name']}")
        print(f"{'#'*80}\n")

        # Clear database before each run
        clear_success, clear_output = clear_database()
        if not clear_success:
            print(f"WARNING: Database clear failed, continuing anyway...")

        # Run the pipeline
        # Use eval mode with CSVs
        cmd = [
            "uv", "run", "python", "-m", pipeline["module"],
            "--file", PDF_FILE,
            "--eval"
        ]
        for csv in CSV_FILES:
            cmd.extend(["--eval-csv", csv])

        success, output = run_command(cmd, f"{pipeline['name']} (eval mode)", timeout=600)

        # Look for generated eval JSON
        eval_json = pipeline["eval_json_pattern"].format(stem=Path(PDF_FILE).stem)

        write_results(pipeline["name"], success, output, eval_json)

    print(f"\n{'#'*80}")
    print(f"# Evaluation Complete!")
    print(f"# Results written to: {RESULTS_FILE}")
    print(f"{'#'*80}\n")


if __name__ == "__main__":
    main()
