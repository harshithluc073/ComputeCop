"""Background endpoint health watching and circuit breaker state."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from computecop.logging import get_logger, log_event
from computecop.models import utc_now

if TYPE_CHECKING:
    from computecop.endpoints import EndpointCapabilityRegistry
    from computecop.residency import ModelResidencyTracker
    from computecop.upstream import UpstreamRouter


class CircuitBreakerState(str, Enum):
    """Circuit breaker lifecycle state for an upstream endpoint."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True, slots=True)
class CircuitBreakerStatus:
    """Snapshot of circuit breaker state for observability and routing."""

    state: CircuitBreakerState
    consecutive_failures: int
    failure_threshold: int
    cooldown_seconds: float
    opened_at: datetime | None
    half_open_at: datetime | None
    last_failure_at: datetime | None
    last_success_at: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "opened_at": self.opened_at.isoformat() if self.opened_at is not None else None,
            "half_open_at": (
                self.half_open_at.isoformat() if self.half_open_at is not None else None
            ),
            "last_failure_at": (
                self.last_failure_at.isoformat() if self.last_failure_at is not None else None
            ),
            "last_success_at": (
                self.last_success_at.isoformat() if self.last_success_at is not None else None
            ),
            "allows_traffic": self.state is not CircuitBreakerState.OPEN,
        }


class EndpointCircuitBreaker:
    """Per-endpoint circuit breaker with closed, open, and half-open states."""

    def __init__(
        self,
        *,
        failure_threshold: int,
        cooldown_seconds: float,
        half_open_successes: int = 1,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        if cooldown_seconds <= 0:
            raise ValueError("cooldown_seconds must be positive")
        if half_open_successes < 1:
            raise ValueError("half_open_successes must be at least 1")
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._half_open_successes = half_open_successes
        self._state = CircuitBreakerState.CLOSED
        self._consecutive_failures = 0
        self._half_open_success_count = 0
        self._opened_at_monotonic: float | None = None
        self._half_open_at: datetime | None = None
        self._opened_at: datetime | None = None
        self._last_failure_at: datetime | None = None
        self._last_success_at: datetime | None = None

    @property
    def state(self) -> CircuitBreakerState:
        self._maybe_advance_from_open()
        return self._state

    def snapshot(self) -> CircuitBreakerStatus:
        self._maybe_advance_from_open()
        return CircuitBreakerStatus(
            state=self._state,
            consecutive_failures=self._consecutive_failures,
            failure_threshold=self._failure_threshold,
            cooldown_seconds=self._cooldown_seconds,
            opened_at=self._opened_at,
            half_open_at=self._half_open_at,
            last_failure_at=self._last_failure_at,
            last_success_at=self._last_success_at,
        )

    def allows_traffic(self) -> bool:
        self._maybe_advance_from_open()
        return self._state is not CircuitBreakerState.OPEN

    def record_success(self) -> CircuitBreakerStatus:
        now = utc_now()
        self._last_success_at = now
        if self._state is CircuitBreakerState.HALF_OPEN:
            self._half_open_success_count += 1
            if self._half_open_success_count >= self._half_open_successes:
                self._reset_to_closed()
        else:
            self._consecutive_failures = 0
            self._state = CircuitBreakerState.CLOSED
            self._opened_at = None
            self._opened_at_monotonic = None
            self._half_open_at = None
            self._half_open_success_count = 0
        return self.snapshot()

    def record_failure(self) -> CircuitBreakerStatus:
        now = utc_now()
        self._last_failure_at = now
        self._consecutive_failures += 1
        if self._state is CircuitBreakerState.HALF_OPEN:
            self._trip_open(now)
        elif self._consecutive_failures >= self._failure_threshold:
            self._trip_open(now)
        return self.snapshot()

    def _trip_open(self, now: datetime) -> None:
        self._state = CircuitBreakerState.OPEN
        self._opened_at = now
        self._opened_at_monotonic = time.monotonic()
        self._half_open_at = None
        self._half_open_success_count = 0

    def _reset_to_closed(self) -> None:
        self._state = CircuitBreakerState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        self._opened_at_monotonic = None
        self._half_open_at = None
        self._half_open_success_count = 0

    def _maybe_advance_from_open(self) -> None:
        if self._state is not CircuitBreakerState.OPEN or self._opened_at_monotonic is None:
            return
        if time.monotonic() - self._opened_at_monotonic >= self._cooldown_seconds:
            self._state = CircuitBreakerState.HALF_OPEN
            self._half_open_at = utc_now()
            self._half_open_success_count = 0


class CircuitBreakerRegistry:
    """Manage circuit breakers for all configured endpoints."""

    def __init__(
        self,
        endpoint_names: list[str],
        *,
        failure_threshold: int,
        cooldown_seconds: float,
        half_open_successes: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._half_open_successes = half_open_successes
        self._breakers = {name: self._new_breaker() for name in sorted(set(endpoint_names))}

    def for_endpoint(self, name: str) -> EndpointCircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = self._new_breaker()
        return self._breakers[name]

    def snapshot(self, name: str) -> CircuitBreakerStatus:
        return self.for_endpoint(name).snapshot()

    def allows_traffic(self, name: str) -> bool:
        return self.for_endpoint(name).allows_traffic()

    def record_success(self, name: str) -> CircuitBreakerStatus:
        return self.for_endpoint(name).record_success()

    def record_failure(self, name: str) -> CircuitBreakerStatus:
        return self.for_endpoint(name).record_failure()

    def _new_breaker(self) -> EndpointCircuitBreaker:
        return EndpointCircuitBreaker(
            failure_threshold=self._failure_threshold,
            cooldown_seconds=self._cooldown_seconds,
            half_open_successes=self._half_open_successes,
        )


class EndpointHealthWatcher:
    """Background loop that proactively probes endpoint health."""

    def __init__(
        self,
        registry: EndpointCapabilityRegistry,
        router: UpstreamRouter,
        *,
        interval_seconds: float = 15.0,
        jitter_fraction: float = 0.1,
        enabled: bool = True,
        residency_tracker: ModelResidencyTracker | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if jitter_fraction < 0 or jitter_fraction > 0.5:
            raise ValueError("jitter_fraction must be between 0 and 0.5")
        self._registry = registry
        self._router = router
        self._interval_seconds = interval_seconds
        self._jitter_fraction = jitter_fraction
        self._enabled = enabled
        self._residency_tracker = residency_tracker
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._logger = get_logger("health.watcher")

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def start(self) -> None:
        """Start the health watcher if enabled and not already running."""

        if not self._enabled:
            return
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="computecop-health-watcher")

    async def stop(self) -> None:
        """Stop the health watcher and wait for completion."""

        self._stop_event.set()
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def probe_all(self) -> None:
        """Probe every configured endpoint and refresh breaker state."""

        for route in self._router.routes.values():
            await self._registry.probe(route, force=True)

        if self._residency_tracker is not None:
            await self._residency_tracker.update_from_endpoints(
                list(self._router.routes.values()),
                self._router._client,
            )

    async def _run(self) -> None:
        log_event(self._logger, logging.INFO, "endpoint health watcher started")
        try:
            while not self._stop_event.is_set():
                try:
                    await self.probe_all()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log_event(
                        self._logger,
                        logging.ERROR,
                        "endpoint health probe cycle failed",
                        error=repr(exc),
                    )
                delay = self._interval_seconds * (
                    1.0 + random.uniform(-self._jitter_fraction, self._jitter_fraction)
                )
                await asyncio.sleep(max(0.1, delay))
        finally:
            log_event(self._logger, logging.INFO, "endpoint health watcher stopped")
