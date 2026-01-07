Geojit Research CLI

Local, fast RAG over your equity research PDFs with Postgres + Qdrant and the Python AI SDK.

Quick Start
- Requirements: Python 3.13, uv, Postgres running with DB `geojit`, Qdrant running locally (port 6333) or set `QDRANT_URL`.
- API key: put your OpenAI-compatible key in `.env` (single line raw key) or export `OPENAI_API_KEY`.

Install deps
  uv sync

Ingest PDFs
- Place PDFs under `Financial_Research_Agent_Files/` (default) or set `GEOJIT_DATA_DIR`.
- Run ingestion (creates Postgres tables; upserts Qdrant points):
  uv run python main.py ingest

Chat
- One-shot:
  uv run python main.py chat -q "What happened last quarter at Sun Pharma?"
- REPL:
  uv run python main.py chat

Config (env vars)
- `OPENAI_API_KEY` or `.env` (raw key)
- `GEOJIT_MODEL` (default: gpt-5)
- `GEOJIT_EMBED_MODEL` (default: text-embedding-3-large)
- `GEOJIT_DATA_DIR` (default: Financial_Research_Agent_Files)
- `DATABASE_URL` (default: postgresql://localhost/geojit)
- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` (default: geojit)
- `GEOJIT_CHUNK_SIZE`/`GEOJIT_CHUNK_OVERLAP`/`GEOJIT_TOP_K`

Notes
- Answers are grounded only in retrieved PDF chunks and include citations like [Title p.X].
- Multi-turn is supported; follow-ups use chat history, while retrieval uses the current question for speed.
- Designed to scale to 20k+ PDFs: ingestion is batched; retrieval is vector search over Qdrant.

