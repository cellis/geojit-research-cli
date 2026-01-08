# Geojit Research CLI

A financial research assistant that ingests PDFs and provides an AI-powered query interface with deep research capabilities.

## Installation

```bash
# Install dependencies
uv sync

# Set up environment variables
export OPENAI_API_KEY="your-api-key"
export DATABASE_URL="postgresql://user:pass@localhost/geojit"
export QDRANT_URL="http://localhost:6333"
```

## Usage

### Analyst Command

The `analyst` command provides an AI-powered financial research assistant that can answer questions about your ingested research documents.

#### Single Query Mode

Ask a single question and get an immediate answer:

```bash
analyst "what is the revenue for Sun Pharma in Q4?"
analyst "what companies are covered in the research database?"
analyst "what are the sales for Bata India Ltd. in Q4FY24?"
```

#### Interactive Mode

Launch an interactive chat session:

```bash
analyst
```

This starts a REPL where you can ask multiple questions with conversation context:

```
Financial Research Analyst
Ctrl-C to exit | Shift+Tab to toggle deep research mode

💬 NORMAL | You (Shift+Tab to cycle): what is the revenue for Sun Pharma?
Agent: According to the latest data...

💬 NORMAL | You (Shift+Tab to cycle): what about their profit margin?
Agent: Based on the context...
```

#### Deep Research Mode

Deep research mode enables the agent to write and execute Python code and SQL queries to answer complex analytical questions. It has access to:

- **PostgreSQL database** with structured financial data
- **Python execution** for data analysis (pandas, numpy available)
- **SQL query execution** against the geojit database
- 119-second timeout for thorough analysis

Enable deep research mode in three ways:

**1. Via flag:**
```bash
analyst --deep-research "analyze revenue trends for pharma companies"
```

**2. Via keywords in query:**
```bash
analyst "deep research: compare profit margins across sectors"
analyst "think hard about the correlation between EBITDA and revenue"
```

**3. Toggle in interactive mode:**
```
💬 NORMAL | You: toggle
→ Switched to Deep Research mode

🔬 DEEP | You: calculate average revenue growth for tech sector
Agent: <PYTHON>
import psycopg
import pandas as pd
...
```

Type `toggle` or `/toggle` to switch between modes.

#### Database Schema

The agent has access to these PostgreSQL tables:

**companies**
- `id`, `name`, `sector`, `industry`, `metadata`
- Unique companies with sector/industry classification

**financial_metrics**
- `company_id`, `metric_name`, `metric_value`, `period`, `fiscal_year`, `quarter`
- Structured financial data (revenue, profit, EBITDA, margins, etc.)
- Period format: 'Q4FY24', 'FY2023', 'Q1FY25'

**documents**
- `id`, `path`, `title`, `pages`
- PDF metadata

**chunks**
- `document_id`, `text`, `page_start`, `page_end`
- Text chunks for vector search

#### How Deep Research Works

When you ask a complex question, the agent:

1. Searches the vector database (Qdrant) for relevant context
2. Analyzes your question with the database schema
3. Writes Python code or SQL queries to fetch/analyze data
4. Executes the code and interprets results
5. Provides a comprehensive answer with citations

Example workflow:

```bash
analyst --deep-research "what was the average EBITDA margin for pharma companies in Q4FY24?"
```

The agent might:
1. Write SQL to find pharma companies
2. Query financial_metrics for EBITDA_margin in Q4FY24
3. Calculate the average
4. Return the answer with data sources

#### Examples

**Normal mode (fast, context + database-aware):**
```bash
analyst "what companies do you know about?"
# Queries Postgres database directly for available companies

analyst "what is Sun Pharma's latest revenue?"
# Uses vector search + LLM to answer from PDF context
```

**Deep research mode (analytical, code execution):**
```bash
analyst --deep-research "calculate the correlation between revenue and profit margin across all companies"
# Writes Python code to fetch data, calculate correlation, visualize
```

**Interactive with toggle:**
```bash
analyst

💬 NORMAL | You: what companies do we have data for?
Agent: [lists companies from context]

💬 NORMAL | You: toggle
→ Switched to Deep Research mode

🔬 DEEP | You: calculate average revenue for each sector
Agent: <SQL>
SELECT c.sector, AVG(fm.metric_value) as avg_revenue
FROM companies c
JOIN financial_metrics fm ON c.id = fm.company_id
WHERE fm.metric_name = 'revenue'
GROUP BY c.sector;
</SQL>
[results displayed]
```

### Ingesting PDFs

Before using the analyst, ingest your PDF research documents:

```bash
# Ingest all PDFs
analyst --ingest

# Limit to first N files (for testing)
analyst --ingest --max-files 10
```

This will:
1. Parse PDFs and extract text
2. Extract company metadata and financial metrics
3. Store structured data in PostgreSQL
4. Create embeddings and store in Qdrant for semantic search

## Advanced Options

```bash
# Get help
analyst --help

# Single query with deep research
analyst --deep-research "your complex question here"

# Ingest with file limit
analyst --ingest --max-files 5
```

## Configuration

Edit `geojit/config.py` or set environment variables:

- `OPENAI_API_KEY` - OpenAI API key for embeddings and chat
- `DATABASE_URL` - PostgreSQL connection string
- `QDRANT_URL` - Qdrant vector database URL
- `QDRANT_API_KEY` - (Optional) Qdrant API key
- `QDRANT_COLLECTION` - Collection name (default: "geojit_chunks")

## Tips

1. **Start with normal mode** for quick facts and citations
2. **Use deep research** for calculations, comparisons, and analysis
3. **Interactive mode** is great for exploratory research
4. **Toggle mode** mid-conversation when you need deeper analysis
5. **Temporal queries** work automatically ("last quarter" uses today's date)
6. **Keywords** like "deep research" or "think hard" auto-enable deep mode

## Troubleshooting

**"No results found"** - Make sure you've ingested PDFs first:
```bash
analyst --ingest
```

**"Deep research timed out"** - Query is too complex for 119s limit, try breaking it down:
```bash
# Instead of: "analyze everything about all companies"
analyst "what companies do we have?"
analyst --deep-research "analyze revenue for [specific companies]"
```

**SQL errors** - Check database connection and schema:
```bash
psql -d geojit -c "\dt"  # List tables
```
# Geojit Research CLI

This CLI ingests research PDFs, parses structured financial tables, stores metrics in Postgres, and indexes content into Qdrant for fuzzy retrieval. It also includes a chat-style analyst interface for answering questions against the data.

## Setup

1) Environment
- Create `.env` in repo root with `OPENAI_API_KEY=...`
- Optionally set `DATABASE_URL` (defaults to `postgresql://localhost/geojit`)
- Optional Qdrant: set `QDRANT_URL`, `QDRANT_API_KEY`; otherwise defaults to local `127.0.0.1:6333`

2) Python
- Uses Python 3.13+. Prefer `uv` for isolated runs:
  - `uv run -m geojit.ingest_zerox --max-files 1`

3) Node (optional, for Zerox)
- Initialize package.json once: `npm init -y`
- Install Zerox: `npm i zerox`

## Ingestion (Zerox + GPT-5 fallback)

Primary ingestion entry: `geojit/ingest_zerox.py`

Examples:
- Ingest one PDF (writes to Postgres and Qdrant):
  - `uv run -m geojit.ingest_zerox --file Financial_Research_Agent_Files/<file>.pdf`
- Ingest with Zerox only (no fallback):
  - `uv run -m geojit.ingest_zerox --file <file.pdf> --zerox-only`
- Parse only (no DB writes):
  - `uv run -m geojit.ingest_zerox --file <file.pdf> --no-db`

What it stores:
- Postgres tables: `organization`, `profit_loss`, `profit_loss_quarterly`, `balance_sheet`, `change_in_estimate`, `ratios`, `cash_flow`, `shareholding_percentage`, `price_performance`
- Qdrant collections:
  - `${QDRANT_COLLECTION}`: PDF chunk embeddings for retrieval (payload includes `company_name`, `path`, `page_start`, `text`)
  - `${QDRANT_COLLECTION}-companies`: single-vector per company name for fuzzy company lookup (payload includes `company_id`, `organization_id` when available)

## Evaluation (Zerox-only)

Run Zerox-only extraction for a single PDF and evaluate against CSV ground truth:

```
uv run -m geojit.ingest_zerox --eval --file Financial_Research_Agent_Files/SP20241406115209223TTK.pdf \
  --eval-csv "SP20241406115209223TTK - quarterly_profit_loss.csv" \
  --eval-csv "SP20241406115209223TTK - balance_sheet.csv" \
  --eval-csv "SP20241406115209223TTK - change_in_estimates.csv" \
  --eval-csv "SP20241406115209223TTK - profit_loss.csv"
```

Outputs:
- `parsed_output_<pdf-stem>.json` — raw parsed JSON
- `eval_results_<pdf-stem>.json` — evaluation metrics and details

## Analyst (Chat)

Ask questions with `analyst` (installed via `pyproject.toml` entry point):

- Single-turn: `analyst "What’s the revenue last year of HDFI?"`
- Interactive: `analyst`

Resolution strategy:
1) Resolve company name from Postgres first (exact/ILIKE on `companies`/`organization`).
2) If not found, query Qdrant companies collection for nearest name.
3) Retrieve relevant context from Qdrant main collection.
4) LLM answers; for complex queries the coding agent can emit SQL to Postgres tables.

## Notes
- Model: Defaults to `gpt-5` (configurable via `GEOJIT_MODEL`).
- Data directory: `Financial_Research_Agent_Files/` by default (configurable via `GEOJIT_DATA_DIR`).
- No Postgres chunks: Chunks are no longer stored in Postgres; only Qdrant is used for retrieval.
