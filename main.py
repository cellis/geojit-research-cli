import argparse
import sys

from geojit.ingest import ingest_all
from geojit.chat import ChatSession, answer


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

    args = parser.parse_args()

    if args.ingest:
        ingest_all(max_files=args.max_files)
        return 0

    if args.query:
        # Single question mode
        print(answer(args.query, deep_research=args.deep_research))
        return 0

    # REPL mode with deep research toggle
    session = ChatSession()
    deep_research_mode = args.deep_research

    def get_prompt():
        mode = "🔬 DEEP" if deep_research_mode else "💬 NORMAL"
        return f"{mode} | You (Shift+Tab to cycle): "

    print("Financial Research Analyst")
    print("Ctrl-C to exit | Shift+Tab to toggle deep research mode")
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

                resp = answer(q, session=session, deep_research=deep_research_mode)
                print(f"Agent: {resp}\n")

            except EOFError:
                # Handle Ctrl+D
                print()
                return 0

    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
