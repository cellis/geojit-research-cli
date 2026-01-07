from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime

from .config import load_settings

try:
    from ai_sdk import generate_text, openai
    _ai_sdk_available = True
except Exception:
    _ai_sdk_available = False


def execute_python_code(code: str, timeout: int = 119) -> tuple[bool, str]:
    """Execute Python code in a temporary file with timeout."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        result = subprocess.run(
            ['python', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        Path(temp_path).unlink()

        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, f"Error:\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Execution timed out"
    except Exception as e:
        return False, f"Execution error: {str(e)}"


def execute_sql_query(query: str) -> tuple[bool, str | list[dict]]:
    """Execute SQL query against the database."""
    try:
        import psycopg
        import json

        s = load_settings()
        conn = psycopg.connect(s.database_url)

        with conn.cursor() as cur:
            cur.execute(query)

            # Check if it's a SELECT query
            if cur.description:
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
                return True, results
            else:
                conn.commit()
                return True, "Query executed successfully"

    except Exception as e:
        return False, f"SQL error: {str(e)}"


def deep_research(query: str, context: str = "") -> str:
    """
    Use an LLM-powered coding agent to answer complex queries.
    The agent can write Python scripts and SQL queries to analyze data.
    Time-boxed to 119 seconds.
    """
    if not _ai_sdk_available:
        return "Deep research requires ai-sdk-python to be installed."

    s = load_settings()
    start_time = time.time()
    max_duration = 119  # 1 minute 59 seconds

    # Get current date for temporal context
    now = datetime.now()
    date_context = f"Today is {now.strftime('%B %d, %Y')} ({now.strftime('%A')})."

    # Database schema information
    db_schema = """
    Available database tables:

    1. companies (id, name, sector, industry, metadata, created_at, updated_at)
    2. financial_metrics (id, company_id, document_id, metric_name, metric_value,
       metric_value_text, unit, period, quarter, fiscal_year, report_date, metadata)
    3. documents (id, path, sha256, title, pages, created_at)
    4. chunks (id, document_id, chunk_index, text, page_start, page_end, token_count)

    Common metric_name values: 'revenue', 'sales', 'profit', 'net_profit', 'ebitda',
    'ebitda_margin', 'total_assets', 'equity', etc.

    Period format examples: 'Q4FY24', 'FY2023', 'Q1FY25'
    """

    system_prompt = f"""You are a financial research coding agent. {date_context}

{db_schema}

You can:
1. Write Python code to analyze data (it will be executed automatically)
2. Write SQL queries to fetch data from the PostgreSQL database
3. Combine both approaches to answer complex questions

When writing Python code:
- Use standard libraries (psycopg, pandas, numpy are available)
- Print results clearly
- Handle errors gracefully
- Database connection: import psycopg; conn = psycopg.connect("{s.database_url}")

When writing SQL:
- Query the tables above
- Use proper JOINs for multi-table queries
- Format results clearly

Context from vector search:
{context}

Respond with either:
1. <PYTHON>code here</PYTHON> - to execute Python
2. <SQL>query here</SQL> - to execute SQL
3. Plain text if you can answer directly from context

You have {max_duration} seconds total. Work efficiently."""

    from ai_sdk.types import CoreSystemMessage, CoreUserMessage, TextPart

    model = openai(s.openai_model)
    messages = [
        CoreSystemMessage(content=system_prompt),
        CoreUserMessage(content=[TextPart(text=query)])
    ]

    max_iterations = 5
    conversation = []

    for iteration in range(max_iterations):
        elapsed = time.time() - start_time
        if elapsed > max_duration:
            return f"Deep research timed out after {iteration} iterations. Last result: {conversation[-1] if conversation else 'No results'}"

        # Generate response
        resp = generate_text(model=model, messages=messages)
        response_text = resp.text
        conversation.append(f"Agent: {response_text}")

        # Check for code execution requests
        if "<PYTHON>" in response_text and "</PYTHON>" in response_text:
            code_start = response_text.index("<PYTHON>") + 8
            code_end = response_text.index("</PYTHON>")
            code = response_text[code_start:code_end].strip()

            remaining_time = max_duration - (time.time() - start_time)
            success, output = execute_python_code(code, timeout=int(remaining_time))

            result_msg = f"Python execution {'succeeded' if success else 'failed'}:\n{output}"
            conversation.append(f"System: {result_msg}")
            messages.append(CoreUserMessage(content=[TextPart(text=result_msg)]))

        elif "<SQL>" in response_text and "</SQL>" in response_text:
            sql_start = response_text.index("<SQL>") + 5
            sql_end = response_text.index("</SQL>")
            sql = response_text[sql_start:sql_end].strip()

            success, output = execute_sql_query(sql)

            if success and isinstance(output, list):
                import json
                result_msg = f"SQL query succeeded:\n{json.dumps(output, indent=2, default=str)}"
            else:
                result_msg = f"SQL query {'succeeded' if success else 'failed'}:\n{output}"

            conversation.append(f"System: {result_msg}")
            messages.append(CoreUserMessage(content=[TextPart(text=result_msg)]))

        else:
            # Agent provided a direct answer
            return response_text

    return f"Deep research completed {max_iterations} iterations:\n\n" + "\n\n".join(conversation[-3:])
