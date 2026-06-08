"""Lock-safe runtime state for ComputeCop."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import cast

from computecop.models import (
    AdmissionDecision,
    DecisionType,
    QueueLifecycleState,
    SystemState,
    TelemetrySample,
    WorkerState,
    to_jsonable,
)


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """Observed queue worker state."""

    worker_id: str
    state: WorkerState
    active_correlation_id: str | None = None


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
    recent_decisions: tuple[AdmissionDecision, ...] = field(default_factory=tuple)

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
        self._recent_decisions: deque[AdmissionDecision] = deque(maxlen=recent_decision_limit)
        self._decision_by_correlation_id: dict[str, AdmissionDecision] = {}

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

    async def snapshot(self) -> RuntimeSnapshot:
        async with self._lock:
            return RuntimeSnapshot(
                telemetry=self._telemetry,
                system_state=self._system_state,
                global_juice_level=self._global_juice_level,
                yield_active=self._yield_active,
                yield_reason=self._yield_reason,
                queue=self._queue,
                recent_decisions=tuple(self._recent_decisions),
            )