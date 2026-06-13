"""Lock-safe runtime state for ComputeCop."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import cast

from computecop.concurrency import ConcurrencyGovernorSnapshot
from computecop.models import (
    AdmissionDecision,
    DecisionType,
    QueueLifecycleState,
    SystemState,
    TelemetrySample,
    WorkerState,
    to_jsonable,
)
from computecop.residency import ModelResidency, ModelResidencyTracker


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """Observed queue worker state."""

    worker_id: str
    state: WorkerState
    active_correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class EventPersistenceSnapshot:
    """Health of the JSONL event persistence layer."""

    enabled: bool = True
    disabled_reason: str | None = None


@dataclass(frozen=True, slots=True)
class QueueSnapshot:
    """Current queue counters, lifecycle state, and worker observations."""

    lifecycle_state: QueueLifecycleState = QueueLifecycleState.ACCEPTING
    queued: int = 0
    running_background: int = 0
    running_foreground: int = 0
    rejected: int = 0
    completed: int = 0
    drain_deadline_monotonic: float | None = None
    workers: tuple[WorkerSnapshot, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class SchedulerSnapshot:
    """Adaptive scheduler capacity and execution counters."""

    reserved_foreground_slots: int = 4
    effective_foreground_slots: int = 4
    max_background_slots: int = 2
    effective_background_slots: int = 2
    running_foreground: int = 0
    running_background: int = 0
    total_capacity: int = 6
    spare_slots: int = 6
    immediate_executions: int = 0
    queued_executions: int = 0


# Backwards-compatible alias retained for callers that imported the previous name.
QueueCounters = QueueSnapshot


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Dashboard and API friendly runtime snapshot."""

    telemetry: TelemetrySample | None
    system_state: SystemState
    global_juice_level: int
    yield_active: bool
    yield_reason: str | None
    queue: QueueSnapshot
    scheduler: SchedulerSnapshot = field(default_factory=SchedulerSnapshot)
    concurrency: ConcurrencyGovernorSnapshot | None = None
    event_persistence: EventPersistenceSnapshot = field(default_factory=EventPersistenceSnapshot)
    recent_decisions: tuple[AdmissionDecision, ...] = field(default_factory=tuple)
    model_residency: tuple[ModelResidency, ...] = field(default_factory=tuple)
    metrics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], to_jsonable(self))


class RuntimeStateStore:
    """Async lock-protected mutable state."""

    def __init__(self, recent_decision_limit: int = 100) -> None:
        self._lock = asyncio.Lock()
        self._telemetry: TelemetrySample | None = None
        self._system_state = SystemState.NORMAL
        self._global_juice_level = 100
        self._yield_active = False
        self._yield_reason: str | None = None
        self._queue = QueueSnapshot()
        self._scheduler = SchedulerSnapshot()
        self._concurrency: ConcurrencyGovernorSnapshot | None = None
        self._event_persistence = EventPersistenceSnapshot()
        self._recent_decisions: deque[AdmissionDecision] = deque(maxlen=recent_decision_limit)
        self._decision_by_correlation_id: dict[str, AdmissionDecision] = {}
        self.residency_tracker = ModelResidencyTracker()
        self._latency_buckets = (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
        self._request_latency_counts = [0] * (len(self._latency_buckets) + 1)
        self._queue_wait_counts = [0] * (len(self._latency_buckets) + 1)
        self._upstream_duration_counts = [0] * (len(self._latency_buckets) + 1)
        self._total_requests = 0
        self._shaped_requests = 0

    async def update_telemetry(self, telemetry: TelemetrySample) -> None:
        async with self._lock:
            self._telemetry = telemetry

    async def set_policy_state(
        self,
        *,
        system_state: SystemState,
        global_juice_level: int,
        yield_active: bool,
        yield_reason: str | None,
    ) -> None:
        async with self._lock:
            self._system_state = system_state
            self._global_juice_level = max(1, min(100, global_juice_level))
            self._yield_active = yield_active
            self._yield_reason = yield_reason

    async def update_queue(self, queue: QueueSnapshot) -> None:
        async with self._lock:
            self._queue = queue

    async def update_scheduler(self, scheduler: SchedulerSnapshot) -> None:
        async with self._lock:
            self._scheduler = scheduler

    async def update_concurrency(self, concurrency: ConcurrencyGovernorSnapshot) -> None:
        async with self._lock:
            self._concurrency = concurrency

    async def set_event_persistence(self, *, enabled: bool, disabled_reason: str | None) -> None:
        async with self._lock:
            self._event_persistence = EventPersistenceSnapshot(
                enabled=enabled,
                disabled_reason=None if enabled else disabled_reason,
            )

    async def record_decision(self, decision: AdmissionDecision) -> None:
        async with self._lock:
            if len(self._recent_decisions) == self._recent_decisions.maxlen:
                oldest = self._recent_decisions[-1]
                self._decision_by_correlation_id.pop(oldest.correlation_id, None)
            self._recent_decisions.appendleft(decision)
            self._decision_by_correlation_id[decision.correlation_id] = decision
            if decision.decision == DecisionType.REJECT:
                self._queue = QueueSnapshot(
                    lifecycle_state=self._queue.lifecycle_state,
                    queued=self._queue.queued,
                    running_background=self._queue.running_background,
                    running_foreground=self._queue.running_foreground,
                    rejected=self._queue.rejected + 1,
                    completed=self._queue.completed,
                    drain_deadline_monotonic=self._queue.drain_deadline_monotonic,
                    workers=self._queue.workers,
                )

    async def decision_for_correlation_id(self, correlation_id: str) -> AdmissionDecision | None:
        async with self._lock:
            return self._decision_by_correlation_id.get(correlation_id)

    async def record_request_latency(self, latency: float) -> None:
        async with self._lock:
            self._record_value(self._latency_buckets, self._request_latency_counts, latency)

    async def record_queue_wait_time(self, wait_time: float) -> None:
        async with self._lock:
            self._record_value(self._latency_buckets, self._queue_wait_counts, wait_time)

    async def record_upstream_duration(self, duration: float) -> None:
        async with self._lock:
            self._record_value(self._latency_buckets, self._upstream_duration_counts, duration)

    async def record_shaping(self, *, shaped: bool) -> None:
        async with self._lock:
            self._total_requests += 1
            if shaped:
                self._shaped_requests += 1

    def _record_value(self, buckets: tuple[float, ...], counts: list[int], value: float) -> None:
        for idx, bucket in enumerate(buckets):
            if value <= bucket:
                counts[idx] += 1
                return
        counts[-1] += 1

    async def snapshot(self) -> RuntimeSnapshot:
        async with self._lock:
            shaping_ratio = (
                self._shaped_requests / self._total_requests
                if self._total_requests > 0
                else 0.0
            )
            metrics_dict = {
                "request_latency_histogram": {
                    "buckets": list(self._latency_buckets),
                    "counts": list(self._request_latency_counts),
                },
                "queue_wait_time_histogram": {
                    "buckets": list(self._latency_buckets),
                    "counts": list(self._queue_wait_counts),
                },
                "upstream_duration_histogram": {
                    "buckets": list(self._latency_buckets),
                    "counts": list(self._upstream_duration_counts),
                },
                "shaping": {
                    "total_requests": self._total_requests,
                    "shaped_requests": self._shaped_requests,
                    "shaping_ratio": shaping_ratio,
                },
            }
            return RuntimeSnapshot(
                telemetry=self._telemetry,
                system_state=self._system_state,
                global_juice_level=self._global_juice_level,
                yield_active=self._yield_active,
                yield_reason=self._yield_reason,
                queue=self._queue,
                scheduler=self._scheduler,
                concurrency=self._concurrency,
                event_persistence=self._event_persistence,
                recent_decisions=tuple(self._recent_decisions),
                model_residency=tuple(self.residency_tracker.get_estimates()),
                metrics=metrics_dict,
            )
