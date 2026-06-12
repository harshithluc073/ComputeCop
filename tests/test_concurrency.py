from __future__ import annotations

import asyncio

import pytest

from computecop.concurrency import EndpointConcurrencyGovernor
from computecop.config import PolicyConfig
from computecop.models import (
    PolicyRuleEvent,
    PolicyRuleStatus,
    PolicyTrace,
    RequestClass,
    RequestMetadata,
    RequestPriority,
    SystemState,
    TelemetrySample,
    ThermalState,
    utc_now,
)
from computecop.policy import (
    ConcurrencyLimits,
    JuicePolicyEngine,
    PressureReport,
    compute_concurrency_limits,
)


def _pressure_report(
    *,
    system_state: SystemState = SystemState.NORMAL,
    yield_active: bool = False,
    triggered_rules: frozenset[str] = frozenset(),
) -> PressureReport:
    rules = tuple(
        PolicyRuleEvent(
            name=name,
            status=PolicyRuleStatus.TRIGGERED,
            observed=1.0,
            threshold=0.5,
            penalty=10,
            detail=f"{name} triggered",
        )
        for name in triggered_rules
    )
    trace = PolicyTrace(rules=rules, system_state=system_state, summary="test")
    return PressureReport(
        system_state=system_state,
        global_juice_level=70,
        yield_active=yield_active,
        yield_reason="yielding" if yield_active else None,
        reasons=("test",),
        dynamic_yield_percent=85.0,
        dynamic_recover_percent=78.0,
        memory_budget_scale=1.0,
        total_ram_gb=16.0,
        trace=trace,
        concurrency_limits=ConcurrencyLimits(4, 2, 2, 1, ("configured",)),
    )


def _telemetry(
    *,
    ram: float = 50.0,
    swap: float = 0.0,
    thermal: ThermalState = ThermalState.COOL,
) -> TelemetrySample:
    return TelemetrySample(
        timestamp=utc_now(),
        cpu_percent=10.0,
        cpu_per_core_percent=(10.0,),
        ram_total_bytes=16 * 1024**3,
        ram_available_bytes=8 * 1024**3,
        ram_used_percent=ram,
        swap_used_percent=swap,
        disk_read_bytes_per_sec=0.0,
        disk_write_bytes_per_sec=0.0,
        thermal_state=thermal,
    )


def test_compute_concurrency_limits_at_normal_pressure() -> None:
    policy = PolicyConfig(max_foreground_concurrency=4, max_background_concurrency=2)
    limits = compute_concurrency_limits(_pressure_report(), policy)
    assert limits.max_foreground == 4
    assert limits.max_background == 2
    assert limits.max_endpoint_foreground == 2
    assert limits.max_endpoint_background == 1


def test_compute_concurrency_limits_shrink_under_yield() -> None:
    policy = PolicyConfig(max_foreground_concurrency=4, max_background_concurrency=2)
    limits = compute_concurrency_limits(_pressure_report(yield_active=True), policy)
    assert limits.max_background == 0
    assert limits.max_endpoint_background == 0
    assert limits.max_foreground == 2


def test_compute_concurrency_limits_shrink_under_swap_and_thermal() -> None:
    policy = PolicyConfig(max_foreground_concurrency=4, max_background_concurrency=4)
    limits = compute_concurrency_limits(
        _pressure_report(triggered_rules=frozenset({"swap_pressure", "thermal_pressure"})),
        policy,
    )
    assert limits.max_background < 4
    assert limits.max_endpoint_background < 1 or limits.max_endpoint_foreground < 2


def test_policy_evaluate_includes_open_circuit_breaker_shaping() -> None:
    engine = JuicePolicyEngine(PolicyConfig())
    without = engine.evaluate(_telemetry(), open_circuit_breaker_count=0)
    with_breakers = engine.evaluate(_telemetry(), open_circuit_breaker_count=2)
    assert (
        with_breakers.concurrency_limits.max_endpoint_foreground
        < without.concurrency_limits.max_endpoint_foreground
    )


@pytest.mark.asyncio
async def test_endpoint_governor_blocks_when_endpoint_is_at_capacity() -> None:
    governor = EndpointConcurrencyGovernor(["ollama"])
    await governor.update_limits(
        ConcurrencyLimits(
            max_foreground=4,
            max_background=2,
            max_endpoint_foreground=1,
            max_endpoint_background=1,
            reasons=("test",),
        )
    )
    gate = asyncio.Event()

    async def hold_slot() -> None:
        await gate.wait()

    first = asyncio.create_task(
        governor.run_with_capacity("ollama", foreground=True, runner=hold_slot)
    )
    await asyncio.sleep(0.05)
    acquired = asyncio.Event()

    async def quick() -> str:
        acquired.set()
        return "ok"

    second = asyncio.create_task(
        governor.run_with_capacity("ollama", foreground=True, runner=quick)
    )
    await asyncio.sleep(0.1)
    assert not acquired.is_set()
    gate.set()
    assert await asyncio.wait_for(first, timeout=2) is None
    assert await asyncio.wait_for(second, timeout=2) == "ok"


@pytest.mark.asyncio
async def test_endpoint_governor_releases_capacity_on_runner_error() -> None:
    governor = EndpointConcurrencyGovernor(["ollama"])
    await governor.update_limits(
        ConcurrencyLimits(4, 2, 2, 1, ("test",)),
    )

    async def fail() -> None:
        raise RuntimeError("upstream failed")

    with pytest.raises(RuntimeError, match="upstream failed"):
        await governor.run_with_capacity("ollama", foreground=False, runner=fail)

    snapshot = governor.snapshot()
    endpoint = snapshot.endpoints[0]
    assert endpoint.running_foreground == 0
    assert endpoint.running_background == 0


@pytest.mark.asyncio
async def test_endpoint_governor_run_with_capacity_allows_next_request() -> None:
    governor = EndpointConcurrencyGovernor(["ollama"])
    await governor.update_limits(
        ConcurrencyLimits(4, 2, 1, 1, ("test",)),
    )

    async def ok() -> str:
        return "done"

    assert await governor.run_with_capacity("ollama", foreground=True, runner=ok) == "done"
    assert await governor.run_with_capacity("ollama", foreground=True, runner=ok) == "done"


def test_is_foreground_metadata() -> None:
    from computecop.concurrency import is_foreground_metadata

    foreground = RequestMetadata(
        method="POST",
        path="/v1/chat/completions",
        headers={},
        request_class=RequestClass.USER_PROMPT,
        priority=RequestPriority.FOREGROUND,
    )
    background = RequestMetadata(
        method="POST",
        path="/api/chat",
        headers={},
        request_class=RequestClass.BACKGROUND_REQUEST,
        priority=RequestPriority.BACKGROUND,
    )
    assert is_foreground_metadata(foreground) is True
    assert is_foreground_metadata(background) is False
