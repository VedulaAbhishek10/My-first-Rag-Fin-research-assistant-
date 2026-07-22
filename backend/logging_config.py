"""
Logging configuration for the Financial Research Assistant.

Provides:
- JSON-formatted structured logs for production
- Plain-text fallback for development
- Request ID and session ID context propagation
- A `structured()` convenience method on all loggers
"""

import logging
import sys
from contextvars import ContextVar
from typing import Any

# ── Context variables for request tracing ─────────────────────────────────────
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")

# Default format for development (easy to read in a terminal)
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track whether structured logging is enabled
_structured_enabled = False


def setup_logging(log_level: str = "INFO", structured: bool = False) -> None:
    """
    Configure the root logger for the entire application.

    Args:
        log_level:  One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
        structured: If True, use JSON formatting for log aggregation tools.
    """
    global _structured_enabled
    _structured_enabled = structured

    handler = logging.StreamHandler(sys.stdout)

    if structured:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Quiet noisy third-party loggers
    for lib in ("chromadb", "httpx", "sentence_transformers", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    The returned logger has a `structured()` convenience method for
    emitting key-value pairs alongside the log message.

    Args:
        name: Typically __name__ of the calling module.
    """
    logger = logging.getLogger(name)

    if not hasattr(logger, "structured"):

        def structured(level: int, msg: str, **kwargs: Any) -> None:
            if _structured_enabled:
                extra = {"extra_fields": kwargs}
                logger.log(level, msg, extra=extra)
            else:
                # Plain-text fallback: append key=value pairs to the message
                parts = [msg]
                for k, v in kwargs.items():
                    parts.append(f"{k}={v}")
                logger.log(level, " | ".join(parts))

        logger.structured = structured  # type: ignore[attr-defined]

    return logger


def set_request_context(request_id: str, session_id: str = "") -> None:
    """Set request-scoped context for tracing."""
    request_id_var.set(request_id)
    if session_id:
        session_id_var.set(session_id)


def clear_request_context() -> None:
    """Clear request-scoped context."""
    request_id_var.set("")
    session_id_var.set("")


# ── JSON formatter (used when structured=True) ─────────────────────────────────


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for machine ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        req_id = request_id_var.get()
        if req_id:
            entry["request_id"] = req_id
        sess_id = session_id_var.get()
        if sess_id:
            entry["session_id"] = sess_id

        if hasattr(record, "extra_fields"):
            entry.update(record.extra_fields)

        if record.exc_info and record.exc_info[1]:
            entry["exception"] = str(record.exc_info[1])

        return json.dumps(entry, default=str)
