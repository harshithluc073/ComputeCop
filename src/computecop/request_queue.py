"""Priority async request queue."""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Generic, TypeVar

from computecop.config import QueueConfig
from computecop.models import (
    QueueLifecycleState,
    RequestMetadata,
    RequestPriority,
    WorkerState,
)
from computecop.state import QueueSnapshot, WorkerSnapshot

T = TypeVar("T")
QueueChangeCallback = Callable[[QueueSnapshot], Awaitable[None] | None]
CapacityHook = Callable[[RequestMetadata], Awaitable[None]]


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
    cancellation: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def expired(self) -> bool:
        return monotonic() >= self.deadline

    def cancel(self) -> None:
        self.cancelled = True
        self.cancellation.set()
        if not self.future.done():
            self.future.cancel()


class AsyncRequestQueue:
    """Bounded priority queue for background inference requests."""

    def __init__(self, config: QueueConfig) -> None:
        self.config = config
        self._condition = asyncio.Condition()
        self._heap: list[_HeapItem[Any]] = []
        self._sequence = itertools.count()
        self._lifecycle_state = QueueLifecycleState.ACCEPTING
        self._drain_deadline: float | None = None
        self._running_background = 0
        self._running_foreground = 0
        self._completed = 0
        self._rejected = 0
        self._change_callback: QueueChangeCallback | None = None
        self._worker_states: dict[str, WorkerSnapshot] = {}

    def set_change_callback(self, callback: QueueChangeCallback) -> None:
        """Register a callback invoked when queue counters change."""

        self._change_callback = callback

    async def register_worker(self, worker_id: str) -> None:
        """Register a queue worker before its task starts."""

        async with self._condition:
            self._worker_states.setdefault(
                worker_id,
                WorkerSnapshot(worker_id=worker_id, state=WorkerState.IDLE),
            )
        await self._notify_change()

    @property
    def lifecycle_state(self) -> QueueLifecycleState:
        return self._lifecycle_state

    def accepts_background_work(self) -> bool:
        """Return whether new background work may enter the queue."""

        return self._lifecycle_state == QueueLifecycleState.ACCEPTING

    async def pause(self) -> None:
        """Stop accepting new background work without draining existing items."""

        async with self._condition:
            if self._lifecycle_state == QueueLifecycleState.CLOSED:
                return
            self._lifecycle_state = QueueLifecycleState.PAUSED
        await self._notify_change()

    async def resume(self) -> None:
        """Resume accepting background work."""

        async with self._condition:
            if self._lifecycle_state == QueueLifecycleState.CLOSED:
                return
            self._lifecycle_state = QueueLifecycleState.ACCEPTING
            self._drain_deadline = None
            self._condition.notify_all()
        await self._notify_change()

    async def drain(self, deadline: float) -> bool:
        """Stop accepting background work and wait for queued work to finish."""

        async with self._condition:
            if self._lifecycle_state == QueueLifecycleState.CLOSED:
                return True
            self._lifecycle_state = QueueLifecycleState.DRAINING
            self._drain_deadline = deadline
            self._condition.notify_all()
        await self._notify_change()

        while True:
            async with self._condition:
                if self._lifecycle_state == QueueLifecycleState.CLOSED:
                    return True
                if not self._heap and self._running_background == 0:
                    return True
                if monotonic() >= deadline:
                    return False
            await asyncio.sleep(0.05)

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
        rejection_error: QueueFullError | None = None
        async with self._condition:
            rejection_error = self._rejection_for_submit_locked(metadata)
            if rejection_error is None:
                self._discard_expired_locked()
            if rejection_error is None and len(self._heap) >= self.config.max_size:
                self._rejected += 1
                rejection_error = QueueFullError("request queue is full")
            if rejection_error is None:
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
        await self._notify_change()
        if rejection_error is not None:
            raise rejection_error

        try:
            return await future
        except asyncio.CancelledError:
            queued.cancel()
            await self._notify_change()
            raise

    async def get(self) -> QueuedRequest[Any]:
        """Return the next runnable queued request."""

        async with self._condition:
            while True:
                if self._lifecycle_state == QueueLifecycleState.CLOSED and not self._heap:
                    raise QueueFullError("request queue is closed")
                self._discard_expired_locked()
                self._rebalance_aging_locked()
                if self._heap:
                    item = heapq.heappop(self._heap)
                    if item.request.cancelled or item.request.cancellation.is_set():
                        continue
                    if item.request.expired:
                        self._expire_request(item.request)
                        continue
                    self._mark_running(item.request.metadata.priority, delta=1)
                    request = item.request
                    break
                await self._condition.wait()
        await self._notify_change()
        return request

    async def run_worker(
        self,
        worker_id: str,
        *,
        acquire: CapacityHook | None = None,
        release: CapacityHook | None = None,
    ) -> None:
        """Continuously execute queued work until the queue is closed."""

        await self._set_worker_state(worker_id, WorkerState.IDLE)
        while True:
            try:
                await self._set_worker_state(worker_id, WorkerState.IDLE)
                request = await self.get()
            except QueueFullError:
                await self._set_worker_state(worker_id, WorkerState.STOPPED)
                return
            if acquire is not None:
                try:
                    await acquire(request.metadata)
                except asyncio.CancelledError:
                    async with self._condition:
                        self._mark_running(request.metadata.priority, delta=-1)
                    raise
            correlation_id = request.metadata.correlation_id
            try:
                await self._set_worker_state(
                    worker_id,
                    WorkerState.RUNNING,
                    active_correlation_id=correlation_id,
                )
                result = await request.runner()
                if not request.future.done():
                    request.future.set_result(result)
            except Exception as exc:
                await self._set_worker_state(worker_id, WorkerState.FAILED)
                if not request.future.done():
                    request.future.set_exception(exc)
            finally:
                if release is not None:
                    await release(request.metadata)
                async with self._condition:
                    self._mark_running(request.metadata.priority, delta=-1)
                    self._completed += 1
                await self._set_worker_state(worker_id, WorkerState.IDLE)

    async def close(self) -> None:
        """Close the queue and cancel pending work."""

        async with self._condition:
            self._lifecycle_state = QueueLifecycleState.CLOSED
            self._drain_deadline = None
            for item in self._heap:
                item.request.cancel()
            self._heap.clear()
            for worker_id, snapshot in list(self._worker_states.items()):
                if snapshot.state not in {WorkerState.STOPPED, WorkerState.STOPPING}:
                    self._worker_states[worker_id] = WorkerSnapshot(
                        worker_id=worker_id,
                        state=WorkerState.STOPPING,
                        active_correlation_id=snapshot.active_correlation_id,
                    )
            self._condition.notify_all()
        await self._notify_change()

    def snapshot(self) -> QueueSnapshot:
        """Return the current queue snapshot."""

        workers = tuple(sorted(self._worker_states.values(), key=lambda worker: worker.worker_id))
        return QueueSnapshot(
            lifecycle_state=self._lifecycle_state,
            queued=len(self._heap),
            running_background=self._running_background,
            running_foreground=self._running_foreground,
            rejected=self._rejected,
            completed=self._completed,
            drain_deadline_monotonic=self._drain_deadline,
            workers=workers,
        )

    def counters(self) -> QueueSnapshot:
        """Return current queue counters."""

        return self.snapshot()

    def _rejection_for_submit_locked(self, metadata: RequestMetadata) -> QueueFullError | None:
        if self._lifecycle_state == QueueLifecycleState.CLOSED:
            self._rejected += 1
            return QueueFullError("request queue is closed")
        if self._lifecycle_state in {
            QueueLifecycleState.PAUSED,
            QueueLifecycleState.DRAINING,
        } and _is_background_priority(metadata.priority):
            self._rejected += 1
            return QueueFullError("request queue is not accepting background work")
        return None

    def _discard_expired_locked(self) -> None:
        kept: list[_HeapItem[Any]] = []
        for item in self._heap:
            if item.request.cancelled or item.request.cancellation.is_set():
                continue
            if item.request.expired:
                self._expire_request(item.request)
                continue
            kept.append(item)
        if len(kept) != len(self._heap):
            heapq.heapify(kept)
            self._heap = kept

    def _rebalance_aging_locked(self) -> None:
        """Boost long-waiting background items to reduce starvation."""

        interval = self.config.aging_interval_seconds
        if interval <= 0 or not self._heap:
            return
        now = monotonic()
        updated: list[_HeapItem[Any]] = []
        for item in self._heap:
            age_seconds = now - item.request.enqueued_at
            aging_bonus = int(age_seconds / interval)
            base_rank = _priority_rank(item.request.metadata.priority)
            updated.append(
                _HeapItem(
                    priority_rank=max(0, base_rank - aging_bonus),
                    sequence=item.sequence,
                    deadline=item.deadline,
                    request=item.request,
                )
            )
        heapq.heapify(updated)
        self._heap = updated

    @staticmethod
    def _expire_request(request: QueuedRequest[Any]) -> None:
        if not request.future.done():
            request.future.set_exception(QueueTimeoutError("queued request expired"))

    def _mark_running(self, priority: RequestPriority, delta: int) -> None:
        if priority in {RequestPriority.FOREGROUND, RequestPriority.INTERACTIVE}:
            self._running_foreground = max(0, self._running_foreground + delta)
        else:
            self._running_background = max(0, self._running_background + delta)

    async def _set_worker_state(
        self,
        worker_id: str,
        state: WorkerState,
        *,
        active_correlation_id: str | None = None,
    ) -> None:
        async with self._condition:
            self._worker_states[worker_id] = WorkerSnapshot(
                worker_id=worker_id,
                state=state,
                active_correlation_id=active_correlation_id,
            )
        await self._notify_change()

    async def _notify_change(self) -> None:
        if self._change_callback is None:
            return
        with contextlib.suppress(Exception):
            result = self._change_callback(self.snapshot())
            if result is not None:
                await result


def _priority_rank(priority: RequestPriority) -> int:
    ranks = {
        RequestPriority.FOREGROUND: 0,
        RequestPriority.INTERACTIVE: 1,
        RequestPriority.BACKGROUND: 2,
        RequestPriority.BULK: 3,
    }
    return ranks.get(priority, 2)


def _is_background_priority(priority: RequestPriority) -> bool:
    return priority in {RequestPriority.BACKGROUND, RequestPriority.BULK}
