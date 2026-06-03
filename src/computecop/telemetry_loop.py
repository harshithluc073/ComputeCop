"""Asynchronous telemetry loop with smoothing and subscriptions."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable

from computecop.logging import get_logger, log_event
from computecop.models import TelemetrySample
from computecop.telemetry import PsutilTelemetrySampler


TelemetrySubscriber = Callable[[TelemetrySample], Awaitable[None] | None]


class TelemetryLoop:
    """Periodically sample telemetry and publish smoothed snapshots."""

    def __init__(
        self,
        sampler: PsutilTelemetrySampler,
        interval_seconds: float = 1.0,
        smoothing_window: int = 5,
    ) -> None:
        self.sampler = sampler
        self.interval_seconds = interval_seconds
        self.smoothing_window = smoothing_window
        self._history: deque[TelemetrySample] = deque(maxlen=max(1, smoothing_window))
        self._subscribers: list[TelemetrySubscriber] = []
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._logger = get_logger("telemetry.loop")

    def subscribe(self, subscriber: TelemetrySubscriber) -> None:
        """Register a callback invoked for every smoothed telemetry sample."""

        self._subscribers.append(subscriber)

    async def start(self) -> None:
        """Start the telemetry loop if it is not already running."""

        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(), name="computecop-telemetry-loop")

    async def stop(self) -> None:
        """Stop the telemetry loop and wait for completion."""

        self._stop_event.set()
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def wait_stopped(self) -> None:
        """Wait until the loop task exits."""

        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _run(self) -> None:
        log_event(self._logger, logging.INFO, "telemetry loop started")
        try:
            while not self._stop_event.is_set():
                try:
                    raw = await self.sampler.sample()
                    smoothed = self._smooth(raw)
                    await self._publish(smoothed)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log_event(
                        self._logger,
                        logging.ERROR,
                        "telemetry sample failed",
                        error=repr(exc),
                    )
                await asyncio.sleep(self.interval_seconds)
        finally:
            log_event(self._logger, logging.INFO, "telemetry loop stopped")

    def _smooth(self, sample: TelemetrySample) -> TelemetrySample:
        self._history.append(sample)
        if len(self._history) == 1:
            return sample

        cpu = sum(item.cpu_percent for item in self._history) / len(self._history)
        ram = sum(item.ram_used_percent for item in self._history) / len(self._history)
        swap = sum(item.swap_used_percent for item in self._history) / len(self._history)
        read_rate = (
            sum(item.disk_read_bytes_per_sec for item in self._history) / len(self._history)
        )
        write_rate = (
            sum(item.disk_write_bytes_per_sec for item in self._history) / len(self._history)
        )

        return TelemetrySample(
            timestamp=sample.timestamp,
            cpu_percent=cpu,
            cpu_per_core_percent=sample.cpu_per_core_percent,
            ram_total_bytes=sample.ram_total_bytes,
            ram_available_bytes=sample.ram_available_bytes,
            ram_used_percent=ram,
            swap_used_percent=swap,
            disk_read_bytes_per_sec=read_rate,
            disk_write_bytes_per_sec=write_rate,
            thermal_state=sample.thermal_state,
            temperatures=sample.temperatures,
            heavy_processes=sample.heavy_processes,
        )

    async def _publish(self, sample: TelemetrySample) -> None:
        for subscriber in tuple(self._subscribers):
            result = subscriber(sample)
            if result is not None:
                await result
