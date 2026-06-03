"""Thermal sensor detection and fallback heuristics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import psutil

from computecop.models import TemperatureSample, ThermalState


@dataclass(frozen=True, slots=True)
class ThermalThresholds:
    """Temperature thresholds used to classify CPU thermal pressure."""

    warm_celsius: float = 75.0
    hot_celsius: float = 88.0
    critical_celsius: float = 95.0


class ThermalDetector:
    """Classify thermal pressure from psutil sensors with safe fallbacks."""

    def __init__(self, thresholds: ThermalThresholds | None = None) -> None:
        self.thresholds = thresholds or ThermalThresholds()

    def read_temperatures(self) -> tuple[TemperatureSample, ...]:
        """Read available temperature sensors from psutil."""

        sensor_reader = getattr(psutil, "sensors_temperatures", None)
        if sensor_reader is None:
            return ()
        try:
            raw = sensor_reader(fahrenheit=False)
        except (AttributeError, OSError, RuntimeError):
            return ()
        if not raw:
            return ()

        readings: list[TemperatureSample] = []
        for sensor_name, entries in raw.items():
            for entry in entries:
                current = getattr(entry, "current", None)
                if current is None:
                    continue
                label = getattr(entry, "label", "") or sensor_name
                readings.append(
                    TemperatureSample(
                        label=str(label),
                        current_celsius=float(current),
                        high_celsius=_optional_float(getattr(entry, "high", None)),
                        critical_celsius=_optional_float(getattr(entry, "critical", None)),
                    )
                )
        return tuple(readings)

    def classify(
        self,
        temperatures: Iterable[TemperatureSample],
        cpu_percent: float,
        per_core_percent: Iterable[float],
    ) -> ThermalState:
        """Classify thermal state from sensors or Intel laptop-style heuristics."""

        readings = tuple(temperatures)
        if readings:
            hottest = max(sample.current_celsius for sample in readings)
            if (
                any(
                    sample.critical_celsius is not None
                    and sample.current_celsius >= sample.critical_celsius
                    for sample in readings
                )
                or hottest >= self.thresholds.critical_celsius
            ):
                return ThermalState.CRITICAL
            if hottest >= self.thresholds.hot_celsius:
                return ThermalState.HOT
            if hottest >= self.thresholds.warm_celsius:
                return ThermalState.WARM
            return ThermalState.COOL

        cores = tuple(float(value) for value in per_core_percent)
        saturated_cores = sum(1 for value in cores if value >= 92.0)
        if cpu_percent >= 95.0 and saturated_cores >= max(2, len(cores) // 2):
            return ThermalState.HOT
        if cpu_percent >= 88.0 or saturated_cores >= max(1, len(cores) // 3):
            return ThermalState.WARM
        if cpu_percent <= 55.0:
            return ThermalState.COOL
        return ThermalState.UNKNOWN


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        return float(value)
    except ValueError:
        return None
