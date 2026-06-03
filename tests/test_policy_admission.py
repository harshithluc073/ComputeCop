from __future__ import annotations

from computecop.admission import AdmissionController
from computecop.config import PolicyConfig, QueueConfig
from computecop.models import (
    DecisionType,
    RequestClass,
    RequestMetadata,
    RequestPriority,
    TelemetrySample,
    ThermalState,
    utc_now,
)
from computecop.policy import JuicePolicyEngine


def _telemetry(ram: float, cpu: float = 10.0) -> TelemetrySample:
    return TelemetrySample(
        timestamp=utc_now(),
        cpu_percent=cpu,
        cpu_per_core_percent=(cpu,),
        ram_total_bytes=16 * 1024**3,
        ram_available_bytes=2 * 1024**3,
        ram_used_percent=ram,
        swap_used_percent=0.0,
        disk_read_bytes_per_sec=0.0,
        disk_write_bytes_per_sec=0.0,
        thermal_state=ThermalState.COOL,
    )


def test_policy_enters_yield_at_ram_threshold() -> None:
    engine = JuicePolicyEngine(PolicyConfig())
    report = engine.evaluate(_telemetry(90.0))
    assert report.yield_active is True
    assert report.global_juice_level < 70


def test_admission_allows_foreground_during_yield() -> None:
    engine = JuicePolicyEngine(PolicyConfig())
    controller = AdmissionController(engine, QueueConfig())
    report = engine.evaluate(_telemetry(90.0))
    metadata = RequestMetadata(
        method="POST",
        path="/v1/chat/completions",
        headers={},
        request_class=RequestClass.USER_PROMPT,
        priority=RequestPriority.FOREGROUND,
    )
    decision = controller.decide(metadata, report, queue_size=0)
    assert decision.decision == DecisionType.ALLOW


def test_admission_yields_background_during_ram_pressure() -> None:
    engine = JuicePolicyEngine(PolicyConfig())
    controller = AdmissionController(engine, QueueConfig())
    report = engine.evaluate(_telemetry(90.0))
    metadata = RequestMetadata(
        method="POST",
        path="/api/chat",
        headers={},
        request_class=RequestClass.BACKGROUND_REQUEST,
        priority=RequestPriority.BACKGROUND,
    )
    decision = controller.decide(metadata, report, queue_size=0)
    assert decision.decision == DecisionType.YIELD
