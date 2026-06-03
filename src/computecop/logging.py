"""Logging helpers for ComputeCop."""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Mapping
from typing import Any

from rich.logging import RichHandler

from computecop.models import to_jsonable


_CONFIGURED = False


class EventFormatter(logging.Formatter):
    """Compact formatter that appends structured event data as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        event = getattr(record, "event", None)
        if event is None:
            return base
        try:
            encoded = json.dumps(to_jsonable(event), sort_keys=True, separators=(",", ":"))
        except TypeError:
            encoded = json.dumps({"unserializable_event": repr(event)}, sort_keys=True)
        return f"{base} {encoded}"


def configure_logging(level: str = "INFO", rich: bool = True) -> None:
    """Configure root logging exactly once."""

    global _CONFIGURED
    normalized = level.upper()
    if _CONFIGURED:
        logging.getLogger().setLevel(normalized)
        return

    if rich:
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
            log_time_format="[%X]",
        )
        formatter = EventFormatter("%(message)s")
    else:
        handler = logging.StreamHandler(sys.stderr)
        formatter = EventFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    handler.setFormatter(formatter)
    logging.basicConfig(level=normalized, handlers=[handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced ComputeCop logger."""

    return logging.getLogger(name if name.startswith("computecop") else f"computecop.{name}")


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    event: Mapping[str, Any] | None = None,
    **fields: Any,
) -> None:
    """Emit a log record with structured event fields."""

    payload: dict[str, Any] = {}
    if event:
        payload.update(dict(event))
    payload.update(fields)
    logger.log(level, message, extra={"event": payload or None})


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Redact sensitive request headers before logging."""

    sensitive = {"authorization", "cookie", "set-cookie", "x-api-key", "api-key"}
    result: dict[str, str] = {}
    for key, value in headers.items():
        lowered = key.lower()
        result[lowered] = "[redacted]" if lowered in sensitive else value
    return result
