#!/usr/bin/env python3
"""
Agent to extract financial tables from research PDFs.

Orchestrates text-based and vision-based parsing (from geojit.ingest and
geojit.ingest_vision), locates section page ranges heuristically, evaluates
against available CSV ground truth, and saves one JSON per PDF.

Usage examples:
  - Process a single PDF with evaluation if CSVs exist:
      python agent_financial_extraction.py --file Financial_Research_Agent_Files/SP20241406115209223TTK.pdf

  - Process all PDFs under Financial_Research_Agent_Files/ (best-effort):
      python agent_financial_extraction.py --all

Notes:
  - Vision and text strategies rely on LLMs; ensure OPENAI_API_KEY and config are set.
  - If evaluation CSVs are found (based on filename stem), the agent tests multiple
    strategies (text, vision with different DPIs, vision over detected sections) and
    selects the best-performing output. Otherwise, it runs a sensible default strategy.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from geojit.pdf_evaluator import evaluate_parser_output, print_evaluation_report
from geojit.pdf_parser_text import parse_pdf_text_based

# Import internal helpers from vision ingest module
try:
    from geojit.ingest_vision import _pdf_to_images as vision_pdf_to_images  # type: ignore
    from geojit.ingest_vision import _parse_with_gpt5_vision as vision_parse  # type: ignore
    _vision_available = True
except Exception:
    _vision_available = False


# --------------------------------------------------------------------------------------
# Section detection heuristics
# --------------------------------------------------------------------------------------

SECTION_PATTERNS: Dict[str, List[re.Pattern]] = {
    "profit_loss": [
        re.compile(r"profit\s*&?\s*loss", re.I),
        re.compile(r"income\s+statement", re.I),
    ],
    "balance_sheet": [
        re.compile(r"balance\s+sheet", re.I),
    ],
    "cash_flow": [
        re.compile(r"cash\s*flow", re.I),
        re.compile(r"cashflow", re.I),
    ],
    "ratios": [
        re.compile(r"ratios?", re.I),
        re.compile(r"key\s+ratios", re.I),
    ],
    "profit_loss_quarterly": [
        re.compile(r"quarterly\s+(financials\s+)?profit\s*&?\s*loss", re.I),
        re.compile(r"quarterly\s+profit\s*&?\s*loss", re.I),
        re.compile(r"quarterly\s+financials", re.I),
    ],
}


@dataclass
class SectionPages:
    pages: List[int]  # 0-based page indices


def find_section_pages(pdf_path: Path) -> Dict[str, SectionPages]:
    """Scan PDF text to identify likely page indices for each section.

    Heuristic: find the first page that matches a section header pattern and
    extend the range until another section header appears or the document ends.
    """
    doc = fitz.open(str(pdf_path))
    n_pages = len(doc)
    page_texts = []
    try:
        for i in range(n_pages):
            try:
                text = doc.load_page(i).get_text("text") or ""
            except Exception:
                text = ""
            page_texts.append(text)
    finally:
        doc.close()

    # Identify start pages for each section
    starts: List[Tuple[int, str]] = []  # (page_index, section_key)
    for idx, text in enumerate(page_texts):
        t = text[:5000]  # limit scanning for performance
        for section, patterns in SECTION_PATTERNS.items():
            if any(p.search(t) for p in patterns):
                starts.append((idx, section))
    # Deduplicate, keep earliest occurrence per section
    earliest: Dict[str, int] = {}
    for idx, sec in sorted(starts, key=lambda x: x[0]):
        if sec not in earliest:
            earliest[sec] = idx

    # Compute ranges by sorting starts
    ordered = sorted([(p, s) for s, p in earliest.items()], key=lambda x: x[0])
    sections_pages: Dict[str, SectionPages] = {}
    for i, (start_idx, section) in enumerate(ordered):
        end_idx = n_pages - 1
        if i + 1 < len(ordered):
            end_idx = ordered[i + 1][0] - 1
        if start_idx <= end_idx:
            pages = list(range(start_idx, end_idx + 1))
            sections_pages[section] = SectionPages(pages=pages)

    return sections_pages


# --------------------------------------------------------------------------------------
# Logging utilities
# --------------------------------------------------------------------------------------

def setup_logging(log_file: str, quiet: bool) -> logging.Logger:
    logger = logging.getLogger("agent.financial_extraction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING if quiet else logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def log_jsonl(jsonl_path: Optional[Path], event: str, **fields: Any) -> None:
    if not jsonl_path:
        return
    try:
        payload = {"event": event, **fields}
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        # Best-effort logging; don't crash the agent for logging failure
        pass


# --------------------------------------------------------------------------------------
# Extraction strategies
# --------------------------------------------------------------------------------------

def _run_with_timeout(func, timeout_s: Optional[int], *args, **kwargs):
    if timeout_s is None or timeout_s <= 0:
        return func(*args, **kwargs)
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func, *args, **kwargs)
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeout:
            raise TimeoutError(f"Operation timed out after {timeout_s}s")


def strategy_text_full(pdf_path: Path, company_hint: Optional[str], *, allow_network: bool, timeout_s: Optional[int], logger: logging.Logger, jsonl_log: Optional[Path]) -> Optional[dict]:
    if not allow_network:
        logger.info("Skipping text strategy (network disabled)")
        log_jsonl(jsonl_log, "strategy_skipped", name="text_full", reason="network_disabled")
        return None
    try:
        logger.info("Starting text strategy …")
        log_jsonl(jsonl_log, "strategy_start", name="text_full")
        out = _run_with_timeout(parse_pdf_text_based, timeout_s, str(pdf_path), company_hint)
        logger.info("Text strategy done")
        log_jsonl(jsonl_log, "strategy_done", name="text_full", ok=True)
        return out
    except Exception as e:
        logger.warning(f"Text strategy failed: {e}")
        log_jsonl(jsonl_log, "strategy_done", name="text_full", ok=False, error=str(e))
        return None


def pdf_to_images_subset(pdf_path: Path, pages: List[int], dpi: int = 150) -> List[bytes]:
    images: List[bytes] = []
    doc = fitz.open(str(pdf_path))
    try:
        for i in pages:
            if i < 0 or i >= len(doc):
                continue
            page = doc[i]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            images.append(pix.tobytes("png"))
    finally:
        doc.close()
    return images


def strategy_vision_full(pdf_path: Path, company_hint: Optional[str], *, dpi: int, allow_network: bool, timeout_s: Optional[int], logger: logging.Logger, jsonl_log: Optional[Path]) -> Optional[dict]:
    if not _vision_available:
        logger.info("Skipping vision strategy (helpers unavailable)")
        log_jsonl(jsonl_log, "strategy_skipped", name="vision_full", reason="helpers_unavailable")
        return None
    if not allow_network:
        logger.info("Skipping vision strategy (network disabled)")
        log_jsonl(jsonl_log, "strategy_skipped", name="vision_full", reason="network_disabled")
        return None
    try:
        logger.info(f"Starting vision strategy (dpi={dpi}) …")
        log_jsonl(jsonl_log, "strategy_start", name="vision_full", dpi=dpi)
        images = vision_pdf_to_images(pdf_path, dpi=dpi)
        out = _run_with_timeout(vision_parse, timeout_s, images, company_hint=company_hint)
        logger.info("Vision strategy done")
        log_jsonl(jsonl_log, "strategy_done", name="vision_full", dpi=dpi, ok=True)
        return out
    except Exception as e:
        logger.warning(f"Vision strategy failed (dpi={dpi}): {e}")
        log_jsonl(jsonl_log, "strategy_done", name="vision_full", dpi=dpi, ok=False, error=str(e))
        return None


def strategy_vision_sections(pdf_path: Path, company_hint: Optional[str], sections: Dict[str, SectionPages], *, dpi: int, allow_network: bool, timeout_s: Optional[int], logger: logging.Logger, jsonl_log: Optional[Path]) -> Optional[dict]:
    if not _vision_available:
        logger.info("Skipping vision sections strategy (helpers unavailable)")
        log_jsonl(jsonl_log, "strategy_skipped", name="vision_sections", reason="helpers_unavailable")
        return None
    if not allow_network:
        logger.info("Skipping vision sections strategy (network disabled)")
        log_jsonl(jsonl_log, "strategy_skipped", name="vision_sections", reason="network_disabled")
        return None
    # Combine all detected section pages, keeping order and uniqueness
    ordered_pages = []
    seen = set()
    for pages in [sp.pages for _, sp in sorted(sections.items(), key=lambda kv: (kv[1].pages[0] if kv[1].pages else 1e9))]:
        for p in pages:
            if p not in seen:
                seen.add(p)
                ordered_pages.append(p)
    if not ordered_pages:
        return None
    try:
        logger.info(f"Starting vision sections strategy (dpi={dpi}, pages={len(ordered_pages)}) …")
        log_jsonl(jsonl_log, "strategy_start", name="vision_sections", dpi=dpi, pages=ordered_pages)
        images = pdf_to_images_subset(pdf_path, ordered_pages, dpi=dpi)
        out = _run_with_timeout(vision_parse, timeout_s, images, company_hint=company_hint)
        logger.info("Vision sections strategy done")
        log_jsonl(jsonl_log, "strategy_done", name="vision_sections", dpi=dpi, ok=True)
        return out
    except Exception as e:
        logger.warning(f"Vision sections strategy failed (dpi={dpi}): {e}")
        log_jsonl(jsonl_log, "strategy_done", name="vision_sections", dpi=dpi, ok=False, error=str(e))
        return None


# --------------------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------------------

def discover_eval_csvs(pdf_path: Path) -> List[str]:
    """Find eval CSV files for the given PDF by looking for matching stem prefixes."""
    cwd = Path.cwd()
    stem = pdf_path.stem
    candidates = [
        f"{stem} - quarterly_profit_loss.csv",
        f"{stem} - balance_sheet.csv",
        f"{stem} - change_in_estimates.csv",
        f"{stem} - profit_loss.csv",
        f"{stem} - cash_flow.csv",
        f"{stem} - ratios.csv",
    ]
    found = []
    for name in candidates:
        p = cwd / name
        if p.exists():
            found.append(str(p))
    return found


def evaluate(parsed: dict, csv_paths: List[str], logger: logging.Logger, jsonl_log: Optional[Path]) -> dict:
    try:
        res = evaluate_parser_output(parsed, csv_paths)
        log_jsonl(jsonl_log, "evaluation", accuracy=res.get("accuracy", 0.0), total=res.get("total_comparisons", 0))
        return res
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        log_jsonl(jsonl_log, "evaluation", ok=False, error=str(e))
        return {"total_comparisons": 0, "matches": 0, "mismatches": 0, "accuracy": 0.0, "details": []}


# --------------------------------------------------------------------------------------
# Agent Orchestration
# --------------------------------------------------------------------------------------

def process_single_pdf(pdf_path: Path, *, try_text: bool, try_vision: bool, allow_network: bool, timeout_s: Optional[int], logger: logging.Logger, jsonl_log: Optional[Path], output_dir: Path) -> Tuple[Optional[dict], Optional[dict]]:
    """Run multiple strategies on a single PDF. Returns (best_parsed, best_eval)."""
    logger.info(f"Processing {pdf_path.name}")
    log_jsonl(jsonl_log, "process_start", file=str(pdf_path))

    # Inspect to locate sections
    sections = find_section_pages(pdf_path)
    if sections:
        pretty = {k: f"{v.pages[0]+1}-{v.pages[-1]+1}" if v.pages else "-" for k, v in sections.items()}
        logger.info(f"Detected sections: {pretty}")
        log_jsonl(jsonl_log, "sections_detected", sections=pretty)

    # Discover evaluation CSVs (if any)
    csvs = discover_eval_csvs(pdf_path)
    if csvs:
        logger.info(f"Found eval CSVs: {[Path(p).name for p in csvs]}")
        log_jsonl(jsonl_log, "eval_csvs", files=csvs)
    else:
        logger.info("No eval CSVs found; comparisons will be skipped")

    strategies: List[Tuple[str, dict]] = []
    company_hint = pdf_path.stem

    # Run text strategies
    if try_text:
        parsed = strategy_text_full(pdf_path, company_hint, allow_network=allow_network, timeout_s=timeout_s, logger=logger, jsonl_log=jsonl_log)
        if parsed:
            strategies.append(("text_full", parsed))

    # Run vision strategies (if available)
    if try_vision:
        for dpi in (150, 200):
            parsed = strategy_vision_full(pdf_path, company_hint, dpi=dpi, allow_network=allow_network, timeout_s=timeout_s, logger=logger, jsonl_log=jsonl_log)
            if parsed:
                strategies.append((f"vision_full_dpi{dpi}", parsed))
        if sections:
            for dpi in (150, 200):
                parsed = strategy_vision_sections(pdf_path, company_hint, sections, dpi=dpi, allow_network=allow_network, timeout_s=timeout_s, logger=logger, jsonl_log=jsonl_log)
                if parsed:
                    strategies.append((f"vision_sections_dpi{dpi}", parsed))

    if not strategies:
        logger.warning("No strategies produced output")
        log_jsonl(jsonl_log, "process_done", file=str(pdf_path), ok=False, reason="no_output")
        return None, None

    # Evaluate and pick best (if CSVs available)
    best_name = strategies[0][0]
    best_parsed = strategies[0][1]
    best_eval = None
    best_score = -1.0

    if csvs:
        for name, parsed in strategies:
            res = evaluate(parsed, csvs, logger, jsonl_log)
            score = float(res.get("accuracy", 0.0))
            logger.info(f"Strategy {name} accuracy: {score:.2f}%")
            if score > best_score:
                best_score = score
                best_name = name
                best_parsed = parsed
                best_eval = res
        logger.info(f"Selected best strategy: {best_name} ({best_score:.2f}%)")
        log_jsonl(jsonl_log, "best_strategy", name=best_name, score=best_score)
    else:
        # If no evaluation data, prefer vision full (higher dpi) > text
        pref_order = ["vision_full_dpi200", "vision_full_dpi150", "vision_sections_dpi200", "vision_sections_dpi150", "text_full"]
        ordered = sorted(strategies, key=lambda kv: pref_order.index(kv[0]) if kv[0] in pref_order else 999)
        best_name, best_parsed = ordered[0]
        best_eval = None
        logger.info(f"Selected default strategy: {best_name}")
        log_jsonl(jsonl_log, "best_strategy", name=best_name)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / f"parsed_output_agent_{pdf_path.stem}.json"
    try:
        with open(out_json, "w") as f:
            json.dump(best_parsed, f, indent=2, default=str)
        logger.info(f"Saved {out_json}")
    except Exception as e:
        logger.warning(f"Failed to save {out_json}: {e}")

    if best_eval is not None:
        eval_out = output_dir / f"agent_eval_results_{pdf_path.stem}.json"
        try:
            with open(eval_out, "w") as f:
                json.dump(best_eval, f, indent=2, default=str)
            logger.info(f"Saved {eval_out}")
        except Exception as e:
            logger.warning(f"Failed to save {eval_out}: {e}")

    log_jsonl(jsonl_log, "process_done", file=str(pdf_path), ok=True, strategy=best_name)
    return best_parsed, best_eval


def iter_pdfs(root: Path) -> List[Path]:
    return sorted([p for p in root.glob("**/*.pdf")])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Financial table extraction agent")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=str, help="Path to a single PDF to process")
    g.add_argument("--all", action="store_true", help="Process all PDFs in Financial_Research_Agent_Files/")
    parser.add_argument("--no-text", action="store_true", help="Disable text-based strategy")
    parser.add_argument("--no-vision", action="store_true", help="Disable vision-based strategy")
    parser.add_argument("--allow-network", action="store_true", help="Allow network calls (LLM strategies). Default off.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-strategy timeout in seconds (default 180)")
    parser.add_argument("--log-file", type=str, default="agent_financial_extraction.log", help="Path to log file")
    parser.add_argument("--jsonl-log", type=str, default="agent_financial_extraction.jsonl", help="Path to JSONL event log")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to write JSON outputs")
    parser.add_argument("--quiet", action="store_true", help="Reduce console verbosity")
    args = parser.parse_args(argv)

    logger = setup_logging(args.log_file, args.quiet)
    jsonl_path = Path(args.jsonl_log) if args.jsonl_log else None

    try_text = not args.no_text
    try_vision = not args.no_vision
    allow_network = bool(args.allow_network)
    timeout_s: Optional[int] = args.timeout if args.timeout and args.timeout > 0 else None
    output_dir = Path(args.output_dir)

    if args.file:
        p = Path(args.file)
        if not p.exists():
            logger.error(f"File not found: {p}")
            return 2
        process_single_pdf(
            p,
            try_text=try_text,
            try_vision=try_vision,
            allow_network=allow_network,
            timeout_s=timeout_s,
            logger=logger,
            jsonl_log=jsonl_path,
            output_dir=output_dir,
        )
        return 0

    # --all
    root = Path("Financial_Research_Agent_Files")
    if not root.exists():
        logger.error(f"Directory not found: {root}")
        return 2

    pdfs = iter_pdfs(root)
    if not pdfs:
        logger.info(f"No PDFs found under {root}")
        return 0

    total = len(pdfs)
    ok = 0
    for i, p in enumerate(pdfs, 1):
        logger.info(f"[{i}/{total}] {p.name}")
        parsed, _ = process_single_pdf(
            p,
            try_text=try_text,
            try_vision=try_vision,
            allow_network=allow_network,
            timeout_s=timeout_s,
            logger=logger,
            jsonl_log=jsonl_path,
            output_dir=output_dir,
        )
        ok += 1 if parsed else 0
    logger.info(f"Done. Parsed {ok}/{total} PDFs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
