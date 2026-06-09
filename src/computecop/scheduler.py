"""Adaptive scheduler with foreground capacity reservation."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic
from typing import Generic, TypeVar

from computecop.config import PolicyConfig, QueueConfig
from computecop.models import RequestClass, RequestMetadata, RequestPriority, SystemState
from computecop.policy import PressureReport
from computecop.request_queue import AsyncRequestQueue
from computecop.state import SchedulerSnapshot

T = TypeVar("T")
SchedulerChangeCallback = Callable[[SchedulerSnapshot], Awaitable[None] | None]


@dataclass(slots=True)
class ScheduledWork(Generic[T]):
    """A schedulable unit of proxy work."""

    metadata: RequestMetadata
    runner: Callable[[], Awaitable[T]]
    priority: RequestPriority
    enqueued_at: float
    deadline: float
    endpoint_name: str | None
    model: str | None
    request_class: RequestClass
    estimated_cost: int
    cancellation: asyncio.Event


def estimate_work_cost(metadata: RequestMetadata) -> int:
    """Return a lightweight scheduling cost estimate for a request."""

    cost = 1
    if metadata.model:
        cost += min(50, len(metadata.model))
    if metadata.request_class == RequestClass.USER_PROMPT:
        cost += 100
    elif metadata.request_class == RequestClass.BACKGROUND_REQUEST:
        cost += 5
    rank = _priority_rank(metadata.priority)
    return cost + (4 - rank) * 10


def build_scheduled_work(
    metadata: RequestMetadata,
    runner: Callable[[], Awaitable[T]],
    *,
    deadline: float,
) -> ScheduledWork[T]:
    """Construct a scheduled work item from request metadata."""

    return ScheduledWork(
        metadata=metadata,
        runner=runner,
        priority=metadata.priority,
        enqueued_at=monotonic(),
        deadline=deadline,
        endpoint_name=metadata.endpoint_name,
        model=metadata.model,
        request_class=metadata.request_class,
        estimated_cost=estimate_work_cost(metadata),
        cancellation=asyncio.Event(),
    )


class AdaptiveScheduler:
    """Schedule proxy work with foreground reservation and queue aging."""

    def __init__(
        self,
        queue: AsyncRequestQueue,
        *,
        policy_config: PolicyConfig,
        queue_config: QueueConfig,
    ) -> None:
        self.queue = queue
        self.policy_config = policy_config
        self.queue_config = queue_config
        self._condition = asyncio.Condition()
        self._effective_background = policy_config.max_background_concurrency
        self._running_foreground = 0
        self._running_background = 0
        self._immediate_executions = 0
        self._queued_executions = 0
        self._change_callback: SchedulerChangeCallback | None = None
        self._worker_tasks: list[asyncio.Task[None]] = []

    @property
    def total_capacity(self) -> int:
        return (
            self.policy_config.max_foreground_concurrency
            + self.policy_config.max_background_concurrency
        )

    def set_change_callback(self, callback: SchedulerChangeCallback) -> None:
        """Register a callback invoked when scheduler counters change."""

        self._change_callback = callback

    def update_pressure(self, report: PressureReport) -> None:
        """Shrink or restore background capacity based on live pressure."""

        self._effective_background = effective_background_slots(report, self.policy_config)
        self._condition.notify_all()

    def snapshot(self) -> SchedulerSnapshot:
        """Return the current scheduler capacity snapshot."""

        reserved = self.policy_config.max_foreground_concurrency
        total_running = self._running_foreground + self._running_background
        spare_slots = max(0, self.total_capacity - total_running)
        return SchedulerSnapshot(
            reserved_foreground_slots=reserved,
            max_background_slots=self.policy_config.max_background_concurrency,
            effective_background_slots=self._effective_background,
            running_foreground=self._running_foreground,
            running_background=self._running_background,
            total_capacity=self.total_capacity,
            spare_slots=spare_slots,
            immediate_executions=self._immediate_executions,
            queued_executions=self._queued_executions,
        )

    async def start(self) -> None:
        """Start queue workers governed by scheduler capacity."""

        if self._worker_tasks:
            return
        for index in range(self.policy_config.max_background_concurrency):
            worker_id = f"computecop-queue-worker-{index}"
            await self.queue.register_worker(worker_id)
            self._worker_tasks.append(
                asyncio.create_task(
                    self.queue.run_worker(
                        worker_id,
                        acquire=self._acquire_for_metadata,
                        release=self._release_for_metadata,
                    ),
                    name=worker_id,
                )
            )
        await self._notify_change()

    async def stop(self) -> None:
        """Cancel scheduler workers."""

        for task in self._worker_tasks:
            task.cancel()
        for task in self._worker_tasks:
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await task
        self._worker_tasks.clear()
        await self._notify_change()

    async def execute_immediate(
        self,
        metadata: RequestMetadata,
        runner: Callable[[], Awaitable[T]],
    ) -> T:
        """Execute work immediately after acquiring a reserved or spare capacity slot."""

        is_foreground = _is_foreground_priority(metadata.priority)
        await self._acquire_slot(is_foreground)
        self._immediate_executions += 1
        await self._notify_change()
        try:
            return await runner()
        finally:
            self._immediate_executions = max(0, self._immediate_executions - 1)
            await self._release_slot(is_foreground)

    async def execute_queued(
        self,
        metadata: RequestMetadata,
        runner: Callable[[], Awaitable[T]],
        *,
        timeout_seconds: float | None = None,
    ) -> T:
        """Enqueue background work and wait for scheduler-governed execution."""

        self._queued_executions += 1
        await self._notify_change()
        try:
            return await self.queue.submit(metadata, runner, timeout_seconds)
        finally:
            self._queued_executions = max(0, self._queued_executions - 1)
            await self._notify_change()

    async def _acquire_for_metadata(self, metadata: RequestMetadata) -> None:
        await self._acquire_slot(_is_foreground_priority(metadata.priority))

    async def _release_for_metadata(self, metadata: RequestMetadata) -> None:
        await self._release_slot(_is_foreground_priority(metadata.priority))

    async def _acquire_slot(self, foreground: bool) -> None:
        async with self._condition:
            while True:
                if foreground:
                    if self._running_foreground < self.policy_config.max_foreground_concurrency:
                        self._running_foreground += 1
                        return
                elif self._can_acquire_background_locked():
                    self._running_background += 1
                    return
                await self._condition.wait()

    async def _release_slot(self, foreground: bool) -> None:
        async with self._condition:
            if foreground:
                self._running_foreground = max(0, self._running_foreground - 1)
            else:
                self._running_background = max(0, self._running_background - 1)
            self._condition.notify_all()
        await self._notify_change()

    def _can_acquire_background_locked(self) -> bool:
        if self._effective_background <= 0:
            return False
        if self._running_background >= self._effective_background:
            return False
        total_running = self._running_foreground + self._running_background
        return total_running < self.total_capacity

    async def _notify_change(self) -> None:
        if self._change_callback is None:
            return
        with contextlib.suppress(Exception):
            result = self._change_callback(self.snapshot())
            if result is not None:
                await result


def effective_background_slots(report: PressureReport, policy_config: PolicyConfig) -> int:
    """Compute the background slot limit for a pressure report."""

    configured = policy_config.max_background_concurrency
    if report.yield_active:
        return 0
    if report.system_state == SystemState.PRESSURED:
        return max(1, configured // 2)
    if report.system_state == SystemState.RECOVERING:
        return max(1, int(configured * 0.75))
    return configured


def _is_foreground_priority(priority: RequestPriority) -> bool:
    return priority in {RequestPriority.FOREGROUND, RequestPriority.INTERACTIVE}


def _priority_rank(priority: RequestPriority) -> int:
    ranks = {
        RequestPriority.FOREGROUND: 0,
        RequestPriority.INTERACTIVE: 1,
        RequestPriority.BACKGROUND: 2,
        RequestPriority.BULK: 3,
    }
    return ranks.get(priority, 2)
