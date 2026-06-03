"""Priority async request queue."""

from __future__ import annotations

import asyncio
import heapq
import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Generic, TypeVar

from computecop.config import QueueConfig
from computecop.models import RequestMetadata, RequestPriority
from computecop.state import QueueCounters


T = TypeVar("T")


class QueueFullError(RuntimeError):
    """Raised when the request queue cannot accept more work."""


class QueueTimeoutError(TimeoutError):
    """Raised when queued work expires before execution."""


@dataclass(order=True)
class _HeapItem(Generic[T]):
    priority_rank: int
    sequence: int
    deadline: float
    request: QueuedRequest[T] = field(compare=False)


@dataclass(slots=True)
class QueuedRequest(Generic[T]):
    """Queued async unit of proxy work."""

    metadata: RequestMetadata
    runner: Callable[[], Awaitable[T]]
    enqueued_at: float
    deadline: float
    future: asyncio.Future[T]
    cancelled: bool = False

    @property
    def expired(self) -> bool:
        return monotonic() >= self.deadline

    def cancel(self) -> None:
        self.cancelled = True
        if not self.future.done():
            self.future.cancel()


class AsyncRequestQueue:
    """Bounded priority queue for background inference requests."""

    def __init__(self, config: QueueConfig) -> None:
        self.config = config
        self._condition = asyncio.Condition()
        self._heap: list[_HeapItem[Any]] = []
        self._sequence = itertools.count()
        self._closed = False
        self._running_background = 0
        self._running_foreground = 0
        self._completed = 0
        self._rejected = 0

    async def submit(
        self,
        metadata: RequestMetadata,
        runner: Callable[[], Awaitable[T]],
        timeout_seconds: float | None = None,
    ) -> T:
        """Submit work and wait for its result."""

        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        deadline = monotonic() + (timeout_seconds or self.config.default_timeout_seconds)
        queued = QueuedRequest(
            metadata=metadata,
            runner=runner,
            enqueued_at=monotonic(),
            deadline=deadline,
            future=future,
        )
        async with self._condition:
            if self._closed:
                self._rejected += 1
                raise QueueFullError("request queue is closed")
            self._discard_expired_locked()
            if len(self._heap) >= self.config.max_size:
                self._rejected += 1
                raise QueueFullError("request queue is full")
            heapq.heappush(
                self._heap,
                _HeapItem(
                    priority_rank=_priority_rank(metadata.priority),
                    sequence=next(self._sequence),
                    deadline=deadline,
                    request=queued,
                ),
            )
            self._condition.notify()

        try:
            return await future
        except asyncio.CancelledError:
            queued.cancel()
            raise

    async def get(self) -> QueuedRequest[Any]:
        """Return the next runnable queued request."""

        async with self._condition:
            while True:
                if self._closed and not self._heap:
                    raise QueueFullError("request queue is closed")
                self._discard_expired_locked()
                if self._heap:
                    item = heapq.heappop(self._heap)
                    if item.request.cancelled:
                        continue
                    if item.request.expired:
                        self._expire_request(item.request)
                        continue
                    self._mark_running(item.request.metadata.priority, delta=1)
                    return item.request
                await self._condition.wait()

    async def run_worker(self) -> None:
        """Continuously execute queued work until the queue is closed."""

        while True:
            request = await self.get()
            try:
                result = await request.runner()
                if not request.future.done():
                    request.future.set_result(result)
            except Exception as exc:
                if not request.future.done():
                    request.future.set_exception(exc)
            finally:
                async with self._condition:
                    self._mark_running(request.metadata.priority, delta=-1)
                    self._completed += 1

    async def close(self) -> None:
        """Close the queue and cancel pending work."""

        async with self._condition:
            self._closed = True
            for item in self._heap:
                item.request.cancel()
            self._heap.clear()
            self._condition.notify_all()

    def counters(self) -> QueueCounters:
        """Return current queue counters."""

        return QueueCounters(
            queued=len(self._heap),
            running_background=self._running_background,
            running_foreground=self._running_foreground,
            rejected=self._rejected,
            completed=self._completed,
        )

    def _discard_expired_locked(self) -> None:
        kept: list[_HeapItem[Any]] = []
        for item in self._heap:
            if item.request.cancelled:
                continue
            if item.request.expired:
                self._expire_request(item.request)
                continue
            kept.append(item)
        if len(kept) != len(self._heap):
            heapq.heapify(kept)
            self._heap = kept

    @staticmethod
    def _expire_request(request: QueuedRequest[Any]) -> None:
        if not request.future.done():
            request.future.set_exception(QueueTimeoutError("queued request expired"))

    def _mark_running(self, priority: RequestPriority, delta: int) -> None:
        if priority in {RequestPriority.FOREGROUND, RequestPriority.INTERACTIVE}:
            self._running_foreground = max(0, self._running_foreground + delta)
        else:
            self._running_background = max(0, self._running_background + delta)


def _priority_rank(priority: RequestPriority) -> int:
    ranks = {
        RequestPriority.FOREGROUND: 0,
        RequestPriority.INTERACTIVE: 1,
        RequestPriority.BACKGROUND: 2,
        RequestPriority.BULK: 3,
    }
    return ranks.get(priority, 2)
