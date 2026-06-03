"""psutil-backed telemetry sampling for ComputeCop."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import psutil

from computecop.models import TelemetrySample, ThermalState, utc_now
from computecop.thermal import ThermalDetector


@dataclass(slots=True)
class DiskCounterSnapshot:
    """Previous disk counter state used for per-second rates."""

    timestamp: float
    read_bytes: int
    write_bytes: int


class PsutilTelemetrySampler:
    """Collect host telemetry using psutil with defensive fallbacks."""

    def __init__(self, thermal_detector: ThermalDetector | None = None) -> None:
        self._disk_snapshot: DiskCounterSnapshot | None = None
        self._thermal_detector = thermal_detector or ThermalDetector()

    async def sample(self) -> TelemetrySample:
        """Collect a telemetry sample without blocking the event loop."""

        return await asyncio.to_thread(self.sample_sync)

    def sample_sync(self) -> TelemetrySample:
        """Collect a telemetry sample synchronously."""

        cpu_percent = float(psutil.cpu_percent(interval=None))
        per_core = tuple(float(value) for value in psutil.cpu_percent(interval=None, percpu=True))
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        read_rate, write_rate = self._disk_rates()
        temperatures = self._thermal_detector.read_temperatures()
        thermal_state = self._thermal_detector.classify(
            temperatures=temperatures,
            cpu_percent=cpu_percent,
            per_core_percent=per_core,
        )

        return TelemetrySample(
            timestamp=utc_now(),
            cpu_percent=cpu_percent,
            cpu_per_core_percent=per_core,
            ram_total_bytes=int(memory.total),
            ram_available_bytes=int(memory.available),
            ram_used_percent=float(memory.percent),
            swap_used_percent=float(swap.percent),
            disk_read_bytes_per_sec=read_rate,
            disk_write_bytes_per_sec=write_rate,
            thermal_state=thermal_state if thermal_state else ThermalState.UNKNOWN,
            temperatures=temperatures,
        )

    def _disk_rates(self) -> tuple[float, float]:
        counters = psutil.disk_io_counters()
        now = time.monotonic()
        if counters is None:
            return 0.0, 0.0

        current = DiskCounterSnapshot(
            timestamp=now,
            read_bytes=int(counters.read_bytes),
            write_bytes=int(counters.write_bytes),
        )
        previous = self._disk_snapshot
        self._disk_snapshot = current
        if previous is None:
            return 0.0, 0.0

        elapsed = max(0.001, current.timestamp - previous.timestamp)
        read_rate = max(0.0, (current.read_bytes - previous.read_bytes) / elapsed)
        write_rate = max(0.0, (current.write_bytes - previous.write_bytes) / elapsed)
        return read_rate, write_rate


def format_bytes_per_second(value: float) -> str:
    """Format a byte-per-second rate for dashboards and diagnostics."""

    units = ["B/s", "KiB/s", "MiB/s", "GiB/s"]
    scaled = float(value)
    for unit in units:
        if abs(scaled) < 1024.0 or unit == units[-1]:
            return f"{scaled:.1f} {unit}"
        scaled /= 1024.0
    return f"{scaled:.1f} GiB/s"
