"""
Structured JSON logging for the AI Insight Agent.

Usage
-----
from app.core.logging import get_logger

logger = get_logger(__name__)
logger.info("model loaded", extra={"model_path": str(path), "run_id": run_id})

Every log record is emitted as a single-line JSON object containing:
    timestamp  – ISO-8601 UTC
    level      – DEBUG | INFO | WARNING | ERROR | CRITICAL
    logger     – dotted module name
    message    – human-readable text
    run_id     – optional correlation ID propagated from the HTTP request
    **kwargs   – any extra= fields passed to the logging call
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional

# Thread-local-like run_id propagation without depending on contextvars in 3.6
# contextvars is available in 3.7+; we target 3.9+.
from contextvars import ContextVar

_run_id: ContextVar[Optional[str]] = ContextVar("run_id", default=None)


def set_run_id(run_id: Optional[str]) -> None:
    """Bind a correlation ID for the current async task."""
    _run_id.set(run_id)


def get_run_id() -> Optional[str]:
    return _run_id.get()


class _JSONFormatter(logging.Formatter):
    """Format every LogRecord as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Base fields always present
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
        }

        # Correlation / run ID — injected from ContextVar or explicit extra
        run_id = record.__dict__.get("run_id") or get_run_id()
        if run_id:
            payload["run_id"] = run_id

        # Merge any caller-supplied extra= fields (skip private logging attrs)
        _logging_internals = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in _logging_internals and not key.startswith("_"):
                payload[key] = value

        # Exception info
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """
    Call once at application startup to replace the root handler with a
    structured JSON handler writing to stdout.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove any existing handlers (e.g. uvicorn's plain-text handler)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "pymongo", "boto3", "botocore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the given name (use __name__ at call site)."""
    return logging.getLogger(name)
