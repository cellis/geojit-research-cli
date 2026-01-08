"""Evaluator for PDF parsers - compares parsed output against ground truth CSV files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def parse_value(value: str) -> float | None:
    """Parse a value from CSV, handling percentages, commas, and empty strings."""
    if not value or value.strip() == '' or value.upper() == 'NA':
        return None

    value = value.strip()

    # Handle percentages
    if value.endswith('%'):
        try:
            return float(value.rstrip('%'))
        except ValueError:
            return None

    # Remove commas and quotes
    value = value.replace(',', '').replace('"', '')

    try:
        return float(value)
    except ValueError:
        return None


def load_csv_ground_truth(csv_path: str) -> dict[str, Any]:
    """Load ground truth data from CSV file."""
    data = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        if not rows:
            return data

        # Determine CSV type based on filename
        csv_name = Path(csv_path).stem.lower()

        if 'quarterly_profit_loss' in csv_name:
            # This CSV mixes quarterly and annual data
            data['type'] = 'mixed'
            data['quarterly_data'] = []
            data['annual_data'] = []

            for row in rows:
                metric_name = row.get('Rs (cr)', row.get('Y.E March (Rs. cr)', ''))
                if not metric_name or 'Growth' in metric_name:
                    continue

                # Separate quarterly and annual periods
                for key, value in row.items():
                    if key in ['Rs (cr)', 'Y.E March (Rs. cr)']:
                        continue

                    # Skip growth columns for now
                    if 'Growth' in key:
                        continue

                    parsed_value = parse_value(value)
                    if parsed_value is None:
                        continue

                    if key.startswith('Q'):  # Quarterly data
                        data['quarterly_data'].append({
                            'metric': metric_name,
                            'period': key,
                            'value': parsed_value
                        })
                    elif key.startswith('FY'):  # Annual data
                        data['annual_data'].append({
                            'metric': metric_name,
                            'period': key,
                            'value': parsed_value
                        })

        elif 'profit_loss' in csv_name and 'quarterly' not in csv_name:
            data['type'] = 'profit_loss'
            data['rows'] = []

            for row in rows:
                metric_name = row.get('Y.E March (Rs. cr)', '')
                if not metric_name or metric_name == '% change':
                    continue

                parsed_row = {'metric': metric_name}
                for key, value in row.items():
                    if key != 'Y.E March (Rs. cr)':
                        parsed_row[key] = parse_value(value)

                data['rows'].append(parsed_row)

        elif 'balance_sheet' in csv_name:
            data['type'] = 'balance_sheet'
            data['rows'] = []

            for row in rows:
                metric_name = row.get('Y.E March (Rs. cr)', '')
                if not metric_name:
                    continue

                parsed_row = {'metric': metric_name}
                for key, value in row.items():
                    if key != 'Y.E March (Rs. cr)':
                        parsed_row[key] = parse_value(value)

                data['rows'].append(parsed_row)

        elif 'change_in_estimates' in csv_name:
            data['type'] = 'change_in_estimate'
            data['rows'] = []

            for row in rows:
                metric_name = row.get('', '')  # First column might be unnamed
                if not metric_name:
                    # Try to find the metric name in the first column
                    for key in row.keys():
                        if row[key] and key.strip() == '':
                            metric_name = row[key]
                            break

                parsed_row = {'metric': metric_name}
                for key, value in row.items():
                    if key != '':
                        parsed_row[key] = parse_value(value)

                data['rows'].append(parsed_row)

    return data


def compare_value(parsed: float | None, expected: float | None, tolerance: float = 0.1) -> dict[str, Any]:
    """
    Compare two values with tolerance.

    Args:
        parsed: Value from parser
        expected: Expected value from ground truth
        tolerance: Percentage tolerance (0.1 = 10%)

    Returns:
        Dictionary with match status and details
    """
    if parsed is None and expected is None:
        return {'match': True, 'reason': 'both_null'}

    if parsed is None:
        return {'match': False, 'reason': 'parsed_is_null', 'expected': expected}

    if expected is None:
        return {'match': False, 'reason': 'expected_is_null', 'parsed': parsed}

    # Check if values are close enough
    if expected == 0:
        # For zero values, use absolute tolerance
        if abs(parsed) < 0.01:
            return {'match': True, 'reason': 'close_to_zero'}
        else:
            return {'match': False, 'reason': 'parsed_not_zero', 'parsed': parsed, 'expected': expected}

    # Percentage difference
    pct_diff = abs((parsed - expected) / expected)

    if pct_diff <= tolerance:
        return {'match': True, 'reason': 'within_tolerance', 'pct_diff': pct_diff * 100}
    else:
        return {
            'match': False,
            'reason': 'outside_tolerance',
            'parsed': parsed,
            'expected': expected,
            'pct_diff': pct_diff * 100
        }


def fuzzy_match_period(parsed_period: str, csv_period: str) -> bool:
    """Check if two periods match, accounting for variations like FY24 vs FY24A."""
    # Normalize both
    p1 = parsed_period.upper().replace('A', '').replace('E', '')
    p2 = csv_period.upper().replace('A', '').replace('E', '')

    # Handle typos like FYZZA -> FY22
    if p2 == 'FYZZ':
        p2 = 'FY22'

    return p1 == p2


def evaluate_parser_output(parsed_data: dict[str, Any], csv_files: list[str]) -> dict[str, Any]:
    """
    Evaluate parser output against ground truth CSV files.

    Args:
        parsed_data: Output from parser
        csv_files: List of paths to CSV ground truth files

    Returns:
        Evaluation results with metrics
    """
    results = {
        'total_comparisons': 0,
        'matches': 0,
        'mismatches': 0,
        'accuracy': 0.0,
        'details': []
    }

    # Map metric name to parser field
    field_mapping = {
        'Sales': 'sales',
        'Revenue': 'revenue',
        'EBITDA': 'ebitda',
        'EBITDA margins': 'ebitda_margin_pct',
        'Depreciation': 'depreciation',
        'EBIT': 'ebit',
        'Interest': 'interest',
        'Other Income': 'other_income',
        'Exceptional Items': 'exceptional_items',
        'PBT': 'pbt',
        'Tax': 'tax',
        'Reported PAT': 'reported_pat',
        'Adj PAT': 'adj_pat',
        'Adj EPS (Rs)': 'adj_eps',
        'No. of Shares': 'shares_outstanding',
        'Cash': 'cash',
        'Accounts Receivable': 'accounts_receivable',
        'Inventories': 'inventories',
        'Other Cur. Assets': 'other_current_assets',
        'Investments': 'investments',
        'Gross Fixed Assets': 'gross_fixed_assets',
        'Net Fixed Assets': 'net_fixed_assets',
        'CWIP': 'cwip',
        'Intangible Assets': 'intangible_assets',
    }

    for csv_file in csv_files:
        ground_truth = load_csv_ground_truth(csv_file)

        if not ground_truth:
            continue

        csv_name = Path(csv_file).stem
        data_type = ground_truth['type']

        # Handle mixed quarterly/annual CSV
        if data_type == 'mixed':
            # Process quarterly data
            for gt_item in ground_truth.get('quarterly_data', []):
                metric_name = gt_item['metric']
                period = gt_item['period']
                expected_value = gt_item['value']

                field_name = field_mapping.get(metric_name)
                if not field_name:
                    continue

                # Find in parsed quarterly data
                parsed_quarterly = parsed_data.get('profit_loss_quarterly', [])
                parsed_value = None
                for item in parsed_quarterly:
                    if item.get('period') == period:
                        parsed_value = item.get(field_name)
                        break

                # Compare
                comparison = compare_value(parsed_value, expected_value)
                results['total_comparisons'] += 1

                if comparison['match']:
                    results['matches'] += 1
                else:
                    results['mismatches'] += 1
                    results['details'].append({
                        'csv_file': csv_name,
                        'metric': metric_name,
                        'field': field_name,
                        'period': period,
                        'parsed': parsed_value,
                        'expected': expected_value,
                        'comparison': comparison
                    })

            # Process annual data (FY24, FY23, etc. from the quarterly CSV)
            for gt_item in ground_truth.get('annual_data', []):
                metric_name = gt_item['metric']
                period = gt_item['period']
                expected_value = gt_item['value']

                field_name = field_mapping.get(metric_name)
                if not field_name:
                    continue

                # Find in parsed annual data
                parsed_annual = parsed_data.get('profit_loss', [])
                parsed_value = None
                for item in parsed_annual:
                    if fuzzy_match_period(item.get('fiscal_year', ''), period):
                        parsed_value = item.get(field_name)
                        break

                # Compare
                comparison = compare_value(parsed_value, expected_value)
                results['total_comparisons'] += 1

                if comparison['match']:
                    results['matches'] += 1
                else:
                    results['mismatches'] += 1
                    results['details'].append({
                        'csv_file': csv_name,
                        'metric': metric_name,
                        'field': field_name,
                        'period': period,
                        'parsed': parsed_value,
                        'expected': expected_value,
                        'comparison': comparison
                    })

        # Handle regular CSV with rows
        elif 'rows' in ground_truth:
            parsed_section = parsed_data.get(data_type, [])

            for gt_row in ground_truth['rows']:
                metric_name = gt_row['metric']
                field_name = field_mapping.get(metric_name)
                if not field_name:
                    continue

                # Compare values for each period
                for period_key, expected_value in gt_row.items():
                    if period_key == 'metric' or expected_value is None:
                        continue

                    # Find corresponding parsed value
                    if data_type in ['profit_loss', 'balance_sheet']:
                        matching_item = None
                        for item in parsed_section:
                            if fuzzy_match_period(item.get('fiscal_year', ''), period_key):
                                matching_item = item
                                break

                        parsed_value = matching_item.get(field_name) if matching_item else None
                    else:
                        continue

                    # Compare
                    comparison = compare_value(parsed_value, expected_value)
                    results['total_comparisons'] += 1

                    if comparison['match']:
                        results['matches'] += 1
                    else:
                        results['mismatches'] += 1
                        results['details'].append({
                            'csv_file': csv_name,
                            'metric': metric_name,
                            'field': field_name,
                            'period': period_key,
                            'parsed': parsed_value,
                            'expected': expected_value,
                            'comparison': comparison
                        })

    # Calculate accuracy
    if results['total_comparisons'] > 0:
        results['accuracy'] = (results['matches'] / results['total_comparisons']) * 100

    return results


def print_evaluation_report(results: dict[str, Any]) -> None:
    """Print a human-readable evaluation report."""
    print("\n" + "="*80)
    print("PDF PARSER EVALUATION REPORT")
    print("="*80)

    print(f"\nTotal Comparisons: {results['total_comparisons']}")
    print(f"Matches: {results['matches']}")
    print(f"Mismatches: {results['mismatches']}")
    print(f"Accuracy: {results['accuracy']:.2f}%")

    if results['mismatches'] > 0:
        print(f"\n{'─'*80}")
        print("MISMATCHES (showing first 20):")
        print(f"{'─'*80}\n")

        for detail in results['details'][:20]:
            if 'metric' in detail:
                print(f"CSV: {detail['csv_file']}")
                print(f"Metric: {detail['metric']} ({detail['field']})")
                print(f"Period: {detail['period']}")
                print(f"Parsed:   {detail['parsed']}")
                print(f"Expected: {detail['expected']}")
                print(f"Reason:   {detail['comparison'].get('reason')}")
                if 'pct_diff' in detail['comparison']:
                    print(f"Diff:     {detail['comparison']['pct_diff']:.2f}%")
                print()

    print("="*80 + "\n")
