import argparse
import sys
import os

from geojit.ingest import ingest_all
from geojit.chat import ChatSession, answer
from geojit.config import load_settings

import threading
import itertools
import time


class _Spinner:
    def __init__(self, message: str = "Working", limit_s: int | None = None):
        self._stop = threading.Event()
        self._thread = None
        self.message = message
        self.limit_s = limit_s
        self._start = None

    def start(self):
        frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        def run():
            self._start = time.time()
            for ch in itertools.cycle(frames):
                if self._stop.is_set():
                    break
                elapsed = int(time.time() - self._start)
                mm = elapsed // 60
                ss = elapsed % 60
                if self.limit_s is not None:
                    lim_mm = self.limit_s // 60
                    lim_ss = self.limit_s % 60
                    timer = f"{mm}:{ss:02d}/{lim_mm}:{lim_ss:02d}"
                else:
                    timer = f"{mm}:{ss:02d}"
                print(f"\r{self.message} {timer} {ch}", end="", flush=True)
                time.sleep(0.08)
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.2)
        # clear line
        print("\r" + " " * 80 + "\r", end="", flush=True)


def _shorten(val: object, n: int = 80) -> str:
    s = str(val)
    return s if len(s) <= n else s[: n - 1] + "…"


def _format_tool_line(t: dict) -> str:
    tool = t.get("tool", "tool")
    if tool == "settings.load":
        return f"- [settings] data_dir={_shorten(t.get('data_dir'))} collection={t.get('qdrant_collection')}"
    if tool == "db.connect":
        if t.get("ok"):
            return f"- [db.connect] ok url={_shorten(t.get('url'))}"
        return f"- [db.connect] failed error={t.get('error')}"
    if tool == "db.resolve_company":
        return (
            f"- [db.resolve_company] candidate={_shorten(t.get('candidate'), 60)} "
            f"match={t.get('match')} source={t.get('source')}"
        )
    if tool == "db.query":
        sql = _shorten(t.get("sql"), 70)
        return f"- [db.query] rows={t.get('rows')} sql=\"{sql}\""
    if tool == "qdrant.retrieve_company":
        return (
            f"- [qdrant.retrieve_company] collection={t.get('collection')} candidate={_shorten(t.get('candidate'), 40)} "
            f"top_k={t.get('top_k')} hits={t.get('hits')} best={t.get('best')}"
        )
    if tool == "qdrant.retrieve":
        titles = t.get("sample_titles") or []
        titles_str = ", ".join(titles[:2])
        return (
            f"- [qdrant.retrieve] collection={t.get('collection')} top_k={t.get('top_k')} "
            f"hits={t.get('hits')} titles={_shorten(titles_str, 50)}"
        )
    if tool == "llm.generate":
        return f"- [llm.generate] provider={t.get('provider')} model={t.get('model')}"
    # Fallback generic formatting
    items = [f"{k}={_shorten(v, 40)}" for k, v in t.items() if k != "tool"]
    return f"- [{tool}] " + " ".join(items)


def _run_with_timeout(func, timeout_s: int, *args, **kwargs):
    done = threading.Event()
    box = {"result": None, "error": None}

    def target():
        try:
            box["result"] = func(*args, **kwargs)
        except Exception as e:
            box["error"] = e
        finally:
            done.set()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    finished = done.wait(timeout_s)
    return finished, box["result"], box["error"]


def main():
    parser = argparse.ArgumentParser(prog="geojit", description="Geojit Research CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest PDFs into Postgres + Qdrant")
    p_ingest.add_argument("--max-files", type=int, default=None, help="Limit number of PDFs for quick test")

    p_chat = sub.add_parser("chat", help="Ask questions against the PDFs")
    p_chat.add_argument("-q", "--query", help="Single-question mode")

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest_all(max_files=args.max_files)
        return 0

    if args.cmd == "chat":
        if args.query:
            print(answer(args.query))
            return 0
        # REPL
        session = ChatSession()
        print("Geojit Research Chat. Ctrl-C to exit.")
        try:
            while True:
                q = input("You: ").strip()
                if not q:
                    continue
                resp = answer(q, session=session)
                print("Agent:", resp)
        except KeyboardInterrupt:
            print()
            return 0

    return 0


def analyst_main():
    """Entry point for the 'analyst' command - simplified chat interface."""
    parser = argparse.ArgumentParser(prog="analyst", description="Financial Research Analyst")
    parser.add_argument("query", nargs="?", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--ingest", action="store_true", help="Ingest PDFs instead of chatting")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of PDFs for ingestion")
    parser.add_argument("--deep-research", action="store_true", help="Enable deep research mode by default")
    parser.add_argument("--db-url", type=str, default=None, help="Override DATABASE_URL for this session (e.g., postgresql://localhost/geojit-1)")

    args = parser.parse_args()

    # Allow per-run database selection for ingestion and chat
    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url

    if args.ingest:
        ingest_all(max_files=args.max_files)
        return 0

    if args.query:
        # Single question mode with hard timeout (1m59s)
        TIMEOUT_S = 119
        sp = _Spinner("Working", limit_s=TIMEOUT_S)
        sp.start()
        try:
            finished, out, err = _run_with_timeout(
                answer, TIMEOUT_S, args.query, None, args.deep_research, True
            )
        finally:
            sp.stop()
        if not finished:
            print(f"Timed out after 1:59. Try simplifying or narrowing the query.")
            return 1
        if err is not None:
            # Also log to error.log with traceback
            try:
                from geojit.error_log import write_error
                import traceback
                write_error("analyst single-shot error", err, context={"query": args.query}, trace=traceback.format_exc())
            except Exception:
                pass
            print(f"Error: {err}")
            return 1
        if isinstance(out, dict):
            print(out.get("text", ""))
            tools = out.get("tools", [])
            if tools:
                print("\nTools:")
                for t in tools:
                    print(_format_tool_line(t))
        else:
            print(out)
        return 0

    # REPL mode with deep research toggle
    session = ChatSession()
    deep_research_mode = args.deep_research
    s = load_settings()

    def get_prompt():
        mode = "🔬 DEEP" if deep_research_mode else "💬 NORMAL"
        return f"{mode} | You (Shift+Tab to cycle): "

    print("Financial Research Analyst")
    print("Ctrl-C to exit | Shift+Tab to toggle deep research mode")
    print(f"Model: {s.openai_model}")
    print()

    try:
        import readline  # For better input handling
    except ImportError:
        pass

    try:
        while True:
            try:
                # Use input with custom prompt
                q = input(get_prompt()).strip()

                # Check for mode toggle (we'll handle this as a command)
                if q.lower() in ['toggle', '/toggle', 'switch', '/switch']:
                    deep_research_mode = not deep_research_mode
                    mode_name = "Deep Research" if deep_research_mode else "Normal"
                    print(f"→ Switched to {mode_name} mode\n")
                    continue

                if not q:
                    continue

                TIMEOUT_S = 119
                sp = _Spinner("Thinking", limit_s=TIMEOUT_S)
                sp.start()
                try:
                    finished, out, err = _run_with_timeout(
                        answer, TIMEOUT_S, q, session, deep_research_mode, True
                    )
                finally:
                    sp.stop()
                if not finished:
                    print("Agent: Timed out after 1:59. Try refining the question.\n")
                    continue
                if err is not None:
                    try:
                        from geojit.error_log import write_error
                        import traceback
                        write_error("analyst repl error", err, context={"query": q}, trace=traceback.format_exc())
                    except Exception:
                        pass
                    print(f"Agent: Error: {err}\n")
                    continue
                if isinstance(out, dict):
                    resp = out.get("text", "")
                    print(f"Agent: {resp}\n")
                    tools = out.get("tools", [])
                    if tools:
                        print("Tools executed:")
                        for t in tools:
                            print(_format_tool_line(t))
                        print()
                else:
                    print(f"Agent: {out}\n")

            except EOFError:
                # Handle Ctrl+D
                print()
                return 0

    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
