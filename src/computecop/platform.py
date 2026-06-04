"""Cross-platform host helpers for ComputeCop."""

from __future__ import annotations

import platform as platform_lib
from dataclasses import dataclass

BYTES_PER_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class HostMemoryProfile:
    """Derived memory characteristics used by policy and telemetry code."""

    total_bytes: int
    minimum_supported_gb: float = 6.0

    @property
    def total_gb(self) -> float:
        return self.total_bytes / BYTES_PER_GIB

    @property
    def meets_minimum(self) -> bool:
        return self.total_gb >= self.minimum_supported_gb

    @property
    def reserved_free_gb(self) -> float:
        """Free RAM reserve used to derive pressure thresholds."""

        total = self.total_gb
        if total <= 8.0:
            return 1.25
        if total <= 16.0:
            return 1.5
        return max(2.0, total * 0.08)

    def dynamic_yield_percent(self, configured_cap_percent: float) -> float:
        """Return the yield threshold for this host."""

        if self.total_bytes <= 0:
            return configured_cap_percent
        reserve_based = 100.0 * (1.0 - (self.reserved_free_gb / max(self.total_gb, 0.1)))
        return max(65.0, min(configured_cap_percent, reserve_based))

    def dynamic_recover_percent(
        self,
        configured_recover_percent: float,
        recover_gap_percent: float,
        configured_yield_percent: float,
    ) -> float:
        """Return the recovery threshold for this host."""

        yield_percent = self.dynamic_yield_percent(configured_yield_percent)
        return max(40.0, min(configured_recover_percent, yield_percent - recover_gap_percent))

    @property
    def budget_scale(self) -> float:
        """Scale token budgets for small-memory machines."""

        if self.total_bytes <= 0:
            return 1.0
        return max(0.5, min(1.0, self.total_gb / 12.0))

    @property
    def heavy_process_pressure_mb(self) -> float:
        """RAM-relative threshold for aggregate heavy process pressure."""

        return max(512.0, self.total_gb * 1024.0 * 0.08)


def current_platform_name() -> str:
    """Return a normalized platform name."""

    system = platform_lib.system().strip().lower()
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return system or "unknown"


def is_windows() -> bool:
    return current_platform_name() == "windows"


def is_macos() -> bool:
    return current_platform_name() == "macos"
