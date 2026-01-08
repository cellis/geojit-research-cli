from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


LOG_PATH = Path("error.log")


def write_error(message: str, exc: BaseException | None = None, context: dict[str, Any] | None = None, trace: str | None = None) -> None:
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"[{ts}] {message}"]
        if context:
            try:
                # shallow stringify to avoid huge logs
                ctx_str = ", ".join(f"{k}={repr(v)[:200]}" for k, v in context.items())
            except Exception:
                ctx_str = str(context)
            lines.append(f"Context: {ctx_str}")
        if exc is not None:
            lines.append(f"Exception: {type(exc).__name__}: {exc}")
        if trace:
            lines.append("Traceback:")
            lines.append(trace)
        elif exc is not None:
            lines.append("Traceback:")
            lines.append(traceback.format_exc())
        lines.append("")
        LOG_PATH.write_text((LOG_PATH.read_text() if LOG_PATH.exists() else "") + "\n".join(lines), encoding="utf-8")
    except Exception:
        # Last-resort: avoid crashing on logging
        pass

