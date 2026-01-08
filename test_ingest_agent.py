#!/usr/bin/env python3
"""Test the agent-based ingestion system."""

from geojit.ingest_agent import ingest_with_agent

if __name__ == "__main__":
    # Test with just 3 PDFs, analyzing first 2
    ingest_with_agent(max_files=3, sample_size=2)
