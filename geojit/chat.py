from __future__ import annotations

from collections import deque
import re
from typing import Deque

from .config import load_settings
from .retriever import retrieve, retrieve_company
from .db import connect
from .embeddings import embed_texts
from .qdrant_store import get_qdrant, search as qdrant_search
from .error_log import write_error

try:
    from ai_sdk import generate_text, openai, tool
    from pydantic import BaseModel, Field
    _ai_sdk_available = True
except Exception:  # pragma: no cover
    _ai_sdk_available = False


class ChatSession:
    def __init__(self, max_turns: int = 20):
        self.history: Deque[dict] = deque(maxlen=max_turns * 2)

    def add_user(self, content: str):
        self.history.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.history.append({"role": "assistant", "content": content})


def _rewrite_query(query: str, session: ChatSession | None) -> str:
    # Lightweight heuristic: if pronouns like "that/it" appear and we have previous user messages,
    # append the last user query for extra context to retrieval.
    if not session or not session.history:
        return query
    qlow = query.lower()
    if any(p in qlow for p in ["that ", "it ", "they ", "last quarter", "last year"]):
        prev = [m["content"] for m in session.history if m.get("role") == "user"]
        if prev:
            return f"{query} (context: {prev[-1]})"
    return query


def _resolve_company_from_postgres(conn, name_query: str) -> dict | None:
    """First try an exact/ILIKE match against companies and organization tables."""
    try:
        with conn.cursor() as cur:
            # Try companies table for exact or ILIKE match
            cur.execute(
                """
                SELECT id, name, sector, industry FROM companies
                WHERE LOWER(name) = LOWER(%s) OR name ILIKE %s
                ORDER BY CASE WHEN LOWER(name) = LOWER(%s) THEN 0 ELSE 1 END, name
                LIMIT 1
                """,
                (name_query, f"%{name_query}%", name_query),
            )
            row = cur.fetchone()
            if row:
                return {
                    "company_id": row[0],
                    "name": row[1],
                    "sector": row[2],
                    "industry": row[3],
                    "source": "companies",
                }

            # Try organization table for a match
            cur.execute(
                """
                SELECT id, name FROM organization
                WHERE LOWER(name) = LOWER(%s) OR name ILIKE %s
                ORDER BY CASE WHEN LOWER(name) = LOWER(%s) THEN 0 ELSE 1 END, name
                LIMIT 1
                """,
                (name_query, f"%{name_query}%", name_query),
            )
            row = cur.fetchone()
            if row:
                return {"organization_id": str(row[0]), "name": row[1], "source": "organization"}
    except Exception:
        pass
    return None


def answer(
    query: str,
    session: ChatSession | None = None,
    deep_research: bool = False,
    return_meta: bool = False,
) -> str | dict:
    s = load_settings()
    tool_log: list[dict] = []
    tool_log.append({"tool": "settings.load", "data_dir": str(s.data_dir), "qdrant_collection": s.qdrant_collection})

    if not _ai_sdk_available:
        raise RuntimeError("ai-sdk-python not available")

    from ai_sdk.types import CoreSystemMessage, CoreUserMessage, CoreAssistantMessage, TextPart
    from datetime import datetime

    # Check if user is requesting deep research
    query_lower = query.lower()
    deep_triggers = ['deep research', 'think hard', 'analyze thoroughly', 'detailed analysis', 'investigate']
    if not deep_research:
        deep_research = any(trigger in query_lower for trigger in deep_triggers)

    # Try to resolve company first via Postgres, then via Qdrant company index as a fallback
    company_hint = None
    try:
        conn = connect(s.database_url)
        tool_log.append({"tool": "db.connect", "ok": True, "url": s.database_url})
    except Exception as e:
        conn = None
        tool_log.append({"tool": "db.connect", "ok": False, "error": type(e).__name__})

    # Define generic tools for the model using ai_sdk's tool factory
    class SqlQueryParams(BaseModel):
        sql: str = Field(description="SQL to execute against Postgres")

    def _exec_sql_query(sql: str) -> dict:
        result: dict = {"columns": [], "rows": []}
        if not conn:
            result["error"] = "no_db_connection"
            tool_log.append({"tool": "tool.invoke", "name": "sql_query", "error": "no_db_connection"})
            print(f"[tool] sql_query -> no_db_connection", flush=True)
            return result
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                cols = [d[0] for d in (cur.description or [])]
                rows = cur.fetchall() if cur.description else []
                # Cap rows for display
                MAX_ROWS = 200
                trimmed = rows[:MAX_ROWS]
                result = {"columns": cols, "rows": trimmed, "truncated": len(rows) > MAX_ROWS}
                tool_log.append({
                    "tool": "tool.invoke",
                    "name": "sql_query",
                    "args": {"sql": sql[:180] + ("…" if len(sql) > 180 else "")},
                    "rows": len(rows),
                    "returned": len(trimmed),
                })
                print(f"[tool] sql_query rows={len(rows)} returned={len(trimmed)}", flush=True)
                return result
        except Exception as e:
            tool_log.append({
                "tool": "tool.invoke",
                "name": "sql_query",
                "args": {"sql": sql[:180] + ("…" if len(sql) > 180 else "")},
                "error": type(e).__name__,
            })
            from .error_log import write_error
            import traceback
            write_error("sql_query failed", e, context={"sql": sql[:500]}, trace=traceback.format_exc())
            print(f"[tool] sql_query error={type(e).__name__}", flush=True)
            return {"error": type(e).__name__}

    class QdrantQueryParams(BaseModel):
        text: str = Field(description="Query text to embed and search")
        top_k: int = Field(default=6, description="Number of results to return")
        collection: str | None = Field(default=None, description="Qdrant collection; defaults to primary")

    def _exec_qdrant_query(text: str, top_k: int = 6, collection: str | None = None) -> list[dict]:
        try:
            if collection and collection != s.qdrant_collection:
                qvec = embed_texts([text], api_key=s.openai_api_key, model=s.embedding_model)[0]
                client = get_qdrant(s.qdrant_url, s.qdrant_api_key)
                hits = qdrant_search(client, collection, qvec.tolist(), top_k=top_k)
                ctxs = []
                for h in hits:
                    p = h.payload or {}
                    ctxs.append({
                        "title": p.get("title"),
                        "page_start": p.get("page_start"),
                        "page_end": p.get("page_end"),
                        "text": (p.get("text") or "")[:400],
                        "path": p.get("path"),
                    })
            else:
                ctxs_full = retrieve(text, top_k=top_k)
                ctxs = [{
                    "title": c.get("title"),
                    "page_start": c.get("page_start"),
                    "page_end": c.get("page_end"),
                    "text": (c.get("text") or "")[:400],
                    "path": c.get("path"),
                } for c in ctxs_full]
            tool_log.append({
                "tool": "tool.invoke",
                "name": "qdrant_query",
                "args": {"text": text[:80] + ("…" if len(text) > 80 else ""), "top_k": top_k, "collection": collection or s.qdrant_collection},
                "result_count": len(ctxs),
            })
            print(f"[tool] qdrant_query top_k={top_k} results={len(ctxs)}", flush=True)
            return ctxs
        except Exception as e:
            tool_log.append({
                "tool": "tool.invoke",
                "name": "qdrant_query",
                "args": {"text": text[:80] + ("…" if len(text) > 80 else ""), "top_k": top_k, "collection": collection or s.qdrant_collection},
                "error": type(e).__name__,
            })
            from .error_log import write_error
            import traceback
            write_error("qdrant_query failed", e, context={"top_k": top_k, "collection": collection or s.qdrant_collection}, trace=traceback.format_exc())
            print(f"[tool] qdrant_query error={type(e).__name__}", flush=True)
            return []

    if conn:
        # Heuristic: look for a capitalized token sequence as a possible company name
        m = None
        # crude: the full query as the lookup key
        candidate = query.strip()
        resolved = _resolve_company_from_postgres(conn, candidate)
        if resolved and resolved.get("name"):
            company_hint = resolved["name"]
            tool_log.append({
                "tool": "db.resolve_company",
                "candidate": candidate,
                "match": company_hint,
                "source": resolved.get("source"),
            })
        else:
            # Try Qdrant companies collection as fallback
            hits = retrieve_company(candidate, top_k=1)
            if hits:
                company_hint = hits[0].get("name")
            tool_log.append({
                "tool": "qdrant.retrieve_company",
                "candidate": candidate,
                "top_k": 1,
                "hits": len(hits or []),
                "collection": f"{s.qdrant_collection}-companies",
                "best": hits[0].get("name") if hits else None,
            })

    # Retrieve context
    r_query = _rewrite_query(query, session)
    contexts = retrieve(r_query, top_k=s.top_k)
    sample_titles = [c.get("title") for c in contexts[:2] if c.get("title")]
    tool_log.append({
        "tool": "qdrant.retrieve",
        "collection": s.qdrant_collection,
        "top_k": s.top_k,
        "hits": len(contexts),
        "sample_titles": sample_titles,
    })
    context_texts = []
    for c in contexts:
        title = c.get("title") or ""
        loc = f"p.{c.get('page_start')}"
        ctx = c.get("text") or ""
        context_texts.append(f"[{title} {loc}]\n{ctx}")
    context_blob = "\n\n---\n\n".join(context_texts)

    # Use deep research mode if requested
    if deep_research:
        from .coding_agent import deep_research as run_deep_research
        return run_deep_research(query, context=context_blob)

    # Get current date/time for temporal grounding
    now = datetime.now()
    date_context = f"Today is {now.strftime('%B %d, %Y')} ({now.strftime('%A')}). Current time: {now.strftime('%I:%M %p')}."

    # Get available companies from database for context
    available_companies = []
    if conn:
        try:
            with conn.cursor() as cur:
                sql_avail = "SELECT DISTINCT name FROM companies ORDER BY name LIMIT 50"
                cur.execute(sql_avail)
                available_companies = [row[0] for row in cur.fetchall()]
                tool_log.append({
                    "tool": "db.query",
                    "sql": sql_avail,
                    "rows": len(available_companies),
                })
        except Exception:
            pass

    companies_context = ""
    if available_companies:
        companies_context = f"\n\nAvailable companies in database: {', '.join(available_companies)}"

    # Database schema information
    db_schema = """
Database Schema (PostgreSQL):

1. companies (id, name, sector, industry, metadata, created_at, updated_at)
   - Primary company registry

2. documents (id, path, sha256, title, pages, created_at)
   - PDF document metadata

3. chunks (id, document_id, chunk_index, text, page_start, page_end, token_count)
   - Text chunks from PDFs (stored in Qdrant for vector search)

4. organization (id UUID, company_id, document_id, name, sector, industry, pdf_path, parsing_method)
   - Organization records linking companies to documents

5. profit_loss (organization_id, fiscal_year, revenue, sales, ebitda, ebitda_margin_pct,
   depreciation, ebit, interest, other_income, pbt, tax, tax_rate, reported_pat, adj_pat,
   eps, adj_eps, shares_outstanding, revenue_growth_pct, ebitda_growth_pct, pat_growth_pct)

6. profit_loss_quarterly (organization_id, period, quarter, fiscal_year, sales, revenue,
   ebitda, ebitda_margin_pct, depreciation, ebit, interest, other_income, exceptional_items,
   pbt, tax, reported_pat, adj_pat, eps, adj_eps, shares_outstanding, revenue_yoy_growth_pct,
   revenue_qoq_growth_pct, ebitda_yoy_growth_pct, ebitda_qoq_growth_pct)

7. balance_sheet (organization_id, fiscal_year, cash, accounts_receivable, inventories,
   other_current_assets, investments, gross_fixed_assets, net_fixed_assets, cwip,
   intangible_assets, total_assets, accounts_payable, short_term_debt, long_term_debt,
   total_liabilities, share_capital, reserves, total_equity, working_capital, net_debt)

8. ratios (organization_id, fiscal_year, roe, roa, roce, current_ratio, quick_ratio,
   debt_to_equity, interest_coverage, pe_ratio, pb_ratio)

9. cash_flow (organization_id, fiscal_year, operating_cash_flow, investing_cash_flow,
   capex, financing_cash_flow, free_cash_flow)

10. change_in_estimate (organization_id, fiscal_year, old_revenue, new_revenue, old_ebitda,
    new_ebitda, old_ebitda_margin_pct, new_ebitda_margin_pct, old_adj_pat, new_adj_pat,
    old_eps, new_eps, revenue_change_pct, ebitda_change_pct, pat_change_pct, eps_change_pct)

11. shareholding_percentage (organization_id, as_of_date, promoter_holding_pct,
    public_holding_pct, institutional_holding_pct, fii_holding_pct, dii_holding_pct,
    retail_holding_pct, pledged_shares_pct)

12. price_performance (organization_id, date, open_price, high_price, low_price, close_price,
    volume, return_1d_pct, return_1w_pct, return_1m_pct, return_3m_pct, return_6m_pct,
    return_1y_pct, return_3y_pct, return_5y_pct, high_52w, low_52w)

Period format examples: 'Q4FY24', 'FY2023', 'Q1FY25'
"""

    sys_prompt = (
        f"{date_context}\n\n"
        "You are a financial research assistant with access to a PostgreSQL database and a Qdrant vector store. "
        f"Your PDFs live under: {s.data_dir}\n"
        f"{companies_context}\n\n"
        f"{db_schema}\n\n"
        "IMPORTANT CONSTRAINTS:\n"
        "- Scope: ONLY use data from Postgres, the Qdrant store, and PDFs in the directory above. Do NOT use external web data.\n"
        "- Citations: When using PDF-derived context, cite as [Title p.X].\n"
        "- Unknowns: If data is not available in DB/Qdrant, say so concisely.\n"
        "- No fabrication: Do NOT invent numbers or sources.\n"
        "- DB-first: For structured financial queries, rely on the schema above.\n"
        "- Time phrases: Resolve 'last year/quarter' using today's date.\n"
        "- Autonomy: Do NOT ask for permission to execute tools or queries; execute what's needed and present the result with sources.\n"
        "- Tool policy: For numeric/database-backed questions (revenue, EBITDA, etc.), perform exactly one sql_query to compute the answer. Only use qdrant_query if the SQL returns no rows; at most one qdrant_query call per question. Do not loop Qdrant calls.\n"
        "- Clarifications: Ask at most one concise clarifying question only if the answer depends on an ambiguity you cannot resolve from the DB/Qdrant context. Otherwise, proceed with sensible defaults.\n"
        "- Defaults: Interpret period labels as stored in DB (e.g., 'FY2020' for fiscal year). If the user says '2020' without 'FY' or 'CY', default to 'FY2020'.\n"
        "- Company scope: Default to companies present in the database. Do not ask to include companies outside the database unless the user explicitly requests it.\n"
    )

    # Build messages using ai_sdk types
    messages = [CoreSystemMessage(content=sys_prompt)]
    if session:
        for msg in session.history:
            if msg["role"] == "user":
                messages.append(CoreUserMessage(content=[TextPart(text=msg["content"])]))
            elif msg["role"] == "assistant":
                messages.append(CoreAssistantMessage(content=[TextPart(text=msg["content"])]))

    # Add current query with context
    user_content = f"Context:\n{context_blob}\n\nQuestion: {query}"
    messages.append(CoreUserMessage(content=[TextPart(text=user_content)]))

    # Generate response using ai_sdk with tool-calling enabled
    try:
        model = openai(s.openai_model)
        # Register tools using the factory API to be compatible across SDK versions
        sql_query_tool = tool(
            name="sql_query",
            description="Execute a SQL query against PostgreSQL and return columns + rows.",
            parameters={
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL to execute against Postgres"}
                },
                "required": ["sql"],
            },
            execute=_exec_sql_query,
        )
        qdrant_query_tool = tool(
            name="qdrant_query",
            description="Vector search Qdrant for relevant PDF chunks.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Query text to embed and search"},
                    "top_k": {"type": "integer", "description": "Number of results to return", "default": 6},
                    "collection": {"type": "string", "description": "Qdrant collection; defaults to primary"},
                },
                "required": ["text"],
            },
            execute=_exec_qdrant_query,
        )
        registered_tools = [sql_query_tool, qdrant_query_tool]
        tool_log.append({
            "tool": "llm.generate",
            "provider": "openai",
            "model": s.openai_model,
            "tools": ["sql_query", "qdrant_query"],
            "messages": {
                "system": sys_prompt[:120] + ("…" if len(sys_prompt) > 120 else ""),
                "user_preview": user_content[:120] + ("…" if len(user_content) > 120 else ""),
            },
        })
        resp = generate_text(model=model, messages=messages, tools=registered_tools)
    except Exception as e:
        write_error("generate_text/tool registration failed", e, context={"model": s.openai_model})
        raise
    text = resp.text
    # Post-process to remove permission-seeking lines like "Shall I fetch that now?"
    def _strip_permission_prompts(t: str) -> str:
        patterns = [
            r"\b[Ss]hall I\b.*\?",
            r"\b[Dd]o you want me to\b.*\?",
            r"\b[Ss]hould I\b.*\?",
            r"\b[Pp]lease confirm\b.*\?",
            r"\b[Cc]an I proceed\b.*\?",
            r"\b[Pp]ermission\b.*\?",
            r"\b[Dd]o you (?:want|need) me to (?:run|query|fetch)\b.*\?",
            r"\b[Ii]s it okay if I\b.*\?",
        ]
        rx = re.compile("(" + ")|(".join(patterns) + ")")
        lines = [ln for ln in t.splitlines() if not rx.search(ln)]
        # Tidy multiple blank lines
        cleaned = []
        for ln in lines:
            if ln.strip() == "" and (len(cleaned) == 0 or cleaned[-1].strip() == ""):
                continue
            cleaned.append(ln)
        return "\n".join(cleaned).strip()

    text = _strip_permission_prompts(text)

    if session:
        session.add_user(query)
        session.add_assistant(text)
    if return_meta:
        return {"text": text, "tools": tool_log, "company_hint": company_hint}
    return text
