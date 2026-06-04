"""RAM pressure yield controller."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from computecop.config import PolicyConfig
from computecop.logging import get_logger, log_event
from computecop.models import SystemState, TelemetrySample
from computecop.platform import HostMemoryProfile

OffloadHook = Callable[[str], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class YieldStatus:
    """Current RAM-yield state."""

    active: bool
    system_state: SystemState
    reason: str | None


class RamYieldController:
    """Manage RAM-pressure yield and recovery hysteresis."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config
        self._active = False
        self._reason: str | None = None
        self._offload_hooks: list[OffloadHook] = []
        self._lock = asyncio.Lock()
        self._logger = get_logger("yield")

    def register_offload_hook(self, hook: OffloadHook) -> None:
        """Register a hook called when yield activates."""

        self._offload_hooks.append(hook)

    async def update(self, telemetry: TelemetrySample) -> YieldStatus:
        """Update yield state from telemetry and call hooks on activation."""

        hooks_to_call: list[OffloadHook] = []
        memory = HostMemoryProfile(
            total_bytes=telemetry.ram_total_bytes,
            minimum_supported_gb=self.config.minimum_supported_ram_gb,
        )
        yield_percent = memory.dynamic_yield_percent(self.config.ram_yield_percent)
        recover_percent = memory.dynamic_recover_percent(
            configured_recover_percent=self.config.ram_recover_percent,
            recover_gap_percent=self.config.ram_recover_gap_percent,
            configured_yield_percent=self.config.ram_yield_percent,
        )
        async with self._lock:
            if not self._active and telemetry.ram_used_percent >= yield_percent:
                self._active = True
                self._reason = (
                    f"RAM usage {telemetry.ram_used_percent:.1f}% >= "
                    f"dynamic yield threshold {yield_percent:.1f}%"
                )
                hooks_to_call = list(self._offload_hooks)
                log_event(self._logger, logging.WARNING, "RAM yield activated", reason=self._reason)
            elif self._active and telemetry.ram_used_percent <= recover_percent:
                log_event(
                    self._logger,
                    logging.INFO,
                    "RAM yield recovered",
                    ram_used_percent=telemetry.ram_used_percent,
                )
                self._active = False
                self._reason = None

            status = self.status_locked()

        for hook in hooks_to_call:
            result = hook(status.reason or "RAM pressure")
            if result is not None:
                await result

        return status

    async def status(self) -> YieldStatus:
        async with self._lock:
            return self.status_locked()

    def status_locked(self) -> YieldStatus:
        return YieldStatus(
            active=self._active,
            system_state=SystemState.YIELDING if self._active else SystemState.NORMAL,
            reason=self._reason,
        )
