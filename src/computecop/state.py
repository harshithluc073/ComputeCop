"""Lock-safe runtime state for ComputeCop."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

from computecop.models import (
    AdmissionDecision,
    DecisionType,
    SystemState,
    TelemetrySample,
    to_jsonable,
)


@dataclass(frozen=True, slots=True)
class QueueCounters:
    """Current queue and worker counters."""

    queued: int = 0
    running_background: int = 0
    running_foreground: int = 0
    rejected: int = 0
    completed: int = 0


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Dashboard and API friendly runtime snapshot."""

    telemetry: TelemetrySample | None
    system_state: SystemState
    global_juice_level: int
    yield_active: bool
    yield_reason: str | None
    queue: QueueCounters
    recent_decisions: tuple[AdmissionDecision, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return to_jsonable(self)


class RuntimeStateStore:
    """Async lock-protected mutable state."""

    def __init__(self, recent_decision_limit: int = 100) -> None:
        self._lock = asyncio.Lock()
        self._telemetry: TelemetrySample | None = None
        self._system_state = SystemState.NORMAL
        self._global_juice_level = 100
        self._yield_active = False
        self._yield_reason: str | None = None
        self._queue = QueueCounters()
        self._recent_decisions: deque[AdmissionDecision] = deque(maxlen=recent_decision_limit)

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

    async def update_queue(self, queue: QueueCounters) -> None:
        async with self._lock:
            self._queue = queue

    async def record_decision(self, decision: AdmissionDecision) -> None:
        async with self._lock:
            self._recent_decisions.appendleft(decision)
            if decision.decision == DecisionType.REJECT:
                self._queue = QueueCounters(
                    queued=self._queue.queued,
                    running_background=self._queue.running_background,
                    running_foreground=self._queue.running_foreground,
                    rejected=self._queue.rejected + 1,
                    completed=self._queue.completed,
                )

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
