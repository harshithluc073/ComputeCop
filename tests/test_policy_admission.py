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


def _telemetry(ram: float, cpu: float = 10.0, total_gb: int = 16) -> TelemetrySample:
    return TelemetrySample(
        timestamp=utc_now(),
        cpu_percent=cpu,
        cpu_per_core_percent=(cpu,),
        ram_total_bytes=total_gb * 1024**3,
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


def test_policy_uses_dynamic_thresholds_for_six_gb_hosts() -> None:
    engine = JuicePolicyEngine(PolicyConfig())
    report = engine.evaluate(_telemetry(80.0, total_gb=6))
    assert report.yield_active is True
    assert report.dynamic_yield_percent < 85.0
    assert report.memory_budget_scale == 0.5


def test_policy_scales_prompt_budget_for_six_gb_hosts() -> None:
    engine = JuicePolicyEngine(PolicyConfig(base_context_tokens=8192, base_output_tokens=2048))
    report = engine.evaluate(_telemetry(50.0, total_gb=6))
    budget = engine.budget_for(RequestClass.USER_PROMPT, report)
    assert budget.max_context_tokens == 4096
    assert budget.max_output_tokens == 1024


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
