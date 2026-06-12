"""Per-endpoint concurrency governor for upstream request forwarding."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from computecop.models import RequestClass, RequestMetadata, RequestPriority
from computecop.policy import ConcurrencyLimits

T = TypeVar("T")
GovernorChangeCallback = Callable[["ConcurrencyGovernorSnapshot"], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class EndpointCapacitySnapshot:
    """Live per-endpoint concurrency usage."""

    endpoint_name: str
    max_foreground: int
    max_background: int
    running_foreground: int
    running_background: int


@dataclass(frozen=True, slots=True)
class ConcurrencyGovernorSnapshot:
    """Snapshot of global limits and per-endpoint capacity usage."""

    limits: ConcurrencyLimits
    endpoints: tuple[EndpointCapacitySnapshot, ...]


class EndpointConcurrencyGovernor:
    """Limit concurrent upstream requests per endpoint and request class."""

    def __init__(self, endpoint_names: list[str]) -> None:
        self._endpoint_names = sorted(set(endpoint_names))
        self._limits = ConcurrencyLimits(
            max_foreground=1,
            max_background=1,
            max_endpoint_foreground=1,
            max_endpoint_background=1,
            reasons=("uninitialized",),
        )
        self._running: dict[str, list[int]] = {name: [0, 0] for name in self._endpoint_names}
        self._condition = asyncio.Condition()
        self._change_callback: GovernorChangeCallback | None = None

    def set_change_callback(self, callback: GovernorChangeCallback) -> None:
        """Register a callback invoked when endpoint capacity counters change."""

        self._change_callback = callback

    async def update_limits(self, limits: ConcurrencyLimits) -> None:
        """Apply new concurrency ceilings from policy."""

        async with self._condition:
            self._limits = limits
            self._condition.notify_all()
        await self._notify_change()

    async def acquire(self, endpoint_name: str, *, foreground: bool) -> None:
        """Acquire a per-endpoint concurrency slot."""

        async with self._condition:
            while not self._can_acquire_locked(endpoint_name, foreground):
                await self._condition.wait()
            counts = self._running.setdefault(endpoint_name, [0, 0])
            if foreground:
                counts[0] += 1
            else:
                counts[1] += 1
        await self._notify_change()

    async def release(self, endpoint_name: str, *, foreground: bool) -> None:
        """Release a previously acquired per-endpoint concurrency slot."""

        async with self._condition:
            counts = self._running.setdefault(endpoint_name, [0, 0])
            if foreground:
                counts[0] = max(0, counts[0] - 1)
            else:
                counts[1] = max(0, counts[1] - 1)
            self._condition.notify_all()
        await self._notify_change()

    async def run_with_capacity(
        self,
        endpoint_name: str,
        *,
        foreground: bool,
        runner: Callable[[], Awaitable[T]],
    ) -> T:
        """Acquire endpoint capacity, run work, and release capacity reliably."""

        await self.acquire(endpoint_name, foreground=foreground)
        try:
            return await runner()
        finally:
            await self.release(endpoint_name, foreground=foreground)

    def snapshot(self) -> ConcurrencyGovernorSnapshot:
        """Return current limits and per-endpoint running counts."""

        endpoints = tuple(
            EndpointCapacitySnapshot(
                endpoint_name=name,
                max_foreground=self._limits.max_endpoint_foreground,
                max_background=self._limits.max_endpoint_background,
                running_foreground=self._running.get(name, [0, 0])[0],
                running_background=self._running.get(name, [0, 0])[1],
            )
            for name in self._endpoint_names
        )
        return ConcurrencyGovernorSnapshot(limits=self._limits, endpoints=endpoints)

    def _can_acquire_locked(self, endpoint_name: str, foreground: bool) -> bool:
        counts = self._running.setdefault(endpoint_name, [0, 0])
        if foreground:
            return counts[0] < self._limits.max_endpoint_foreground
        if self._limits.max_endpoint_background <= 0:
            return False
        return counts[1] < self._limits.max_endpoint_background

    async def _notify_change(self) -> None:
        if self._change_callback is None:
            return
        with contextlib.suppress(Exception):
            result = self._change_callback(self.snapshot())
            if result is not None:
                await result


def is_foreground_metadata(metadata: RequestMetadata) -> bool:
    """Return whether metadata should use foreground endpoint capacity."""

    if metadata.request_class == RequestClass.USER_PROMPT:
        return True
    return metadata.priority in {RequestPriority.FOREGROUND, RequestPriority.INTERACTIVE}
