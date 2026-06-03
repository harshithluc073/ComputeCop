from __future__ import annotations

from types import SimpleNamespace

import pytest

from computecop.config import PolicyConfig
from computecop.models import ThermalState, utc_now
from computecop.processes import HeavyProcessDetector
from computecop.telemetry import PsutilTelemetrySampler
from computecop.thermal import ThermalDetector, ThermalThresholds
from computecop.yielding import RamYieldController


def test_sampler_uses_psutil_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psutil.cpu_percent", lambda interval=None, percpu=False: [10.0, 20.0] if percpu else 15.0)
    monkeypatch.setattr(
        "psutil.virtual_memory",
        lambda: SimpleNamespace(total=16 * 1024**3, available=4 * 1024**3, percent=75.0),
    )
    monkeypatch.setattr("psutil.swap_memory", lambda: SimpleNamespace(percent=5.0))
    monkeypatch.setattr(
        "psutil.disk_io_counters",
        lambda: SimpleNamespace(read_bytes=1000, write_bytes=2000),
    )

    sampler = PsutilTelemetrySampler(
        thermal_detector=ThermalDetector(),
        process_detector=HeavyProcessDetector(limit=0),
    )
    sample = sampler.sample_sync()
    assert sample.cpu_percent == 15.0
    assert sample.ram_used_percent == 75.0
    assert sample.swap_used_percent == 5.0


def test_thermal_detector_classifies_sensor_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = SimpleNamespace(label="package", current=91.0, high=90.0, critical=100.0)
    monkeypatch.setattr("psutil.sensors_temperatures", lambda fahrenheit=False: {"coretemp": [entry]})
    detector = ThermalDetector(ThermalThresholds(warm_celsius=70.0, hot_celsius=85.0, critical_celsius=95.0))
    readings = detector.read_temperatures()
    assert detector.classify(readings, cpu_percent=10.0, per_core_percent=(10.0,)) == ThermalState.HOT


def test_thermal_detector_falls_back_to_cpu_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psutil.sensors_temperatures", lambda fahrenheit=False: {})
    detector = ThermalDetector()
    state = detector.classify((), cpu_percent=96.0, per_core_percent=(95.0, 96.0, 30.0, 20.0))
    assert state == ThermalState.HOT


def test_heavy_process_detector_filters_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    process = SimpleNamespace(
        pid=123,
        info={
            "pid": 123,
            "name": "ollama.exe",
            "cpu_percent": 12.0,
            "memory_info": SimpleNamespace(rss=600 * 1024 * 1024),
            "cmdline": ["ollama", "serve"],
        },
    )
    monkeypatch.setattr("psutil.process_iter", lambda attrs: [process])
    samples = HeavyProcessDetector(limit=5).sample()
    assert len(samples) == 1
    assert samples[0].name == "ollama.exe"


@pytest.mark.asyncio
async def test_ram_yield_controller_hysteresis() -> None:
    controller = RamYieldController(PolicyConfig(ram_yield_percent=85.0, ram_recover_percent=78.0))
    hot = _sample(90.0)
    cool = _sample(70.0)
    assert (await controller.update(hot)).active is True
    assert (await controller.update(_sample(80.0))).active is True
    assert (await controller.update(cool)).active is False


def _sample(ram: float):
    from computecop.models import TelemetrySample

    return TelemetrySample(
        timestamp=utc_now(),
        cpu_percent=10.0,
        cpu_per_core_percent=(10.0,),
        ram_total_bytes=16 * 1024**3,
        ram_available_bytes=4 * 1024**3,
        ram_used_percent=ram,
        swap_used_percent=0.0,
        disk_read_bytes_per_sec=0.0,
        disk_write_bytes_per_sec=0.0,
        thermal_state=ThermalState.COOL,
    )
