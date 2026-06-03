"""Bounded JSONL runtime event persistence."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from computecop.models import to_jsonable, utc_now


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """Persistable runtime event."""

    kind: str
    payload: dict[str, Any]
    timestamp: str


class JsonlEventStore:
    """Small bounded JSONL store for recent ComputeCop events."""

    def __init__(self, path: Path | None = None, max_events: int = 1000) -> None:
        self.path = path or default_event_log_path()
        self.max_events = max_events
        self._lock = asyncio.Lock()

    async def append(self, kind: str, **payload: Any) -> None:
        """Append one event and enforce the retention bound."""

        event = RuntimeEvent(kind=kind, payload=to_jsonable(payload), timestamp=utc_now().isoformat())
        encoded = json.dumps(to_jsonable(event), sort_keys=True)
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.write("\n")
            await asyncio.to_thread(self._trim_sync)

    async def tail(self, limit: int = 100) -> tuple[dict[str, Any], ...]:
        """Return recent events."""

        async with self._lock:
            if not self.path.exists():
                return ()
            lines = await asyncio.to_thread(self.path.read_text, encoding="utf-8")
        rows = [line for line in lines.splitlines() if line.strip()]
        parsed: list[dict[str, Any]] = []
        for line in rows[-limit:]:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                parsed.append(value)
        return tuple(parsed)

    def _trim_sync(self) -> None:
        if not self.path.exists():
            return
        rows = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) <= self.max_events:
            return
        self.path.write_text("\n".join(rows[-self.max_events :]) + "\n", encoding="utf-8")


def default_event_log_path() -> Path:
    """Return the platform-appropriate default event log path."""

    base = os.getenv("LOCALAPPDATA")
    if base:
        return Path(base) / "ComputeCop" / "events.jsonl"
    return Path.home() / ".cache" / "computecop" / "events.jsonl"
