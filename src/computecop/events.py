"""Bounded JSONL runtime event persistence."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from computecop.models import to_jsonable, utc_now

PersistenceCallback = Callable[[bool, str | None], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """Persistable runtime event."""

    kind: str
    payload: dict[str, Any]
    timestamp: str


class JsonlEventStore:
    """Small bounded JSONL store for recent ComputeCop events.

    Persistence is best effort: a single unwritable event path disables further
    writes instead of crashing the runtime. The store records why persistence was
    disabled and notifies an optional callback so operators can be warned.
    """

    def __init__(self, path: Path | None = None, max_events: int = 1000) -> None:
        self.path = path or default_event_log_path()
        self.max_events = max_events
        self._lock = asyncio.Lock()
        self._persistence_disabled = False
        self._disabled_reason: str | None = None
        self._on_persistence_change: PersistenceCallback | None = None

    @property
    def persistence_disabled(self) -> bool:
        """Return whether event persistence has been disabled by a write failure."""

        return self._persistence_disabled

    @property
    def disabled_reason(self) -> str | None:
        """Return the reason persistence was disabled, if any."""

        return self._disabled_reason

    def set_persistence_callback(self, callback: PersistenceCallback | None) -> None:
        """Register a callback invoked when persistence is enabled or disabled."""

        self._on_persistence_change = callback

    async def append(self, kind: str, **payload: Any) -> None:
        """Append one event, enforce retention, and survive write failures."""

        event = RuntimeEvent(
            kind=kind, payload=to_jsonable(payload), timestamp=utc_now().isoformat()
        )
        encoded = json.dumps(to_jsonable(event), sort_keys=True)
        async with self._lock:
            if self._persistence_disabled:
                return
            try:
                await asyncio.to_thread(self._write_sync, encoded)
            except OSError as exc:
                await self._disable_persistence(_describe_os_error(exc))

    async def tail(self, limit: int = 100) -> tuple[dict[str, Any], ...]:
        """Return the most recent retained events."""

        return await self.read_events(limit=limit)

    async def read_events(self, limit: int | None = None) -> tuple[dict[str, Any], ...]:
        """Return retained events in chronological order, optionally tail-limited."""

        async with self._lock:
            if not self.path.exists():
                return ()
            text = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        return _parse_rows(text, limit)

    def _write_sync(self, encoded: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._trim_sync()

    async def _disable_persistence(self, reason: str) -> None:
        self._persistence_disabled = True
        self._disabled_reason = reason
        callback = self._on_persistence_change
        if callback is not None:
            await callback(False, reason)

    def _trim_sync(self) -> None:
        if not self.path.exists():
            return
        rows = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) <= self.max_events:
            return
        self.path.write_text("\n".join(rows[-self.max_events :]) + "\n", encoding="utf-8")


def _parse_rows(text: str, limit: int | None) -> tuple[dict[str, Any], ...]:
    rows = [line for line in text.splitlines() if line.strip()]
    if limit is not None:
        rows = rows[-limit:]
    parsed: list[dict[str, Any]] = []
    for line in rows:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed.append(value)
    return tuple(parsed)


def _describe_os_error(exc: OSError) -> str:
    detail = exc.strerror or str(exc)
    if exc.filename:
        return f"{detail}: {exc.filename}"
    return detail


def default_event_log_path() -> Path:
    """Return the platform-appropriate default event log path."""

    base = os.getenv("LOCALAPPDATA")
    if base:
        return Path(base) / "ComputeCop" / "events.jsonl"
    return Path.home() / ".cache" / "computecop" / "events.jsonl"
