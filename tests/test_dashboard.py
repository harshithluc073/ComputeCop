from __future__ import annotations

from unittest.mock import AsyncMock

from rich.console import Console, Group

from computecop.concurrency import (
    ConcurrencyGovernorSnapshot,
    EndpointCapacitySnapshot,
)
from computecop.dashboard import Dashboard
from computecop.dashboard_panels import (
    DEFAULT_LAYOUT,
    render_decision_panel,
    render_endpoint_panel,
    render_header,
    render_policy_panel,
    render_resource_panel,
    render_scheduler_panel,
    render_trace_panel,
    render_worker_panel,
)
from computecop.endpoints import (
    EndpointCapabilities,
    EndpointHealthStatus,
    EndpointRecord,
    EndpointRoutingMetadata,
)
from computecop.models import (
    AdmissionDecision,
    DecisionType,
    EndpointKind,
    JuiceBudget,
    PolicyRuleEvent,
    PolicyRuleStatus,
    PolicyTrace,
    QueueLifecycleState,
    RequestClass,
    RequestPriority,
    SystemState,
    TelemetrySample,
    ThermalState,
    WorkerState,
    utc_now,
)
from computecop.policy import ConcurrencyLimits
from computecop.state import (
    QueueSnapshot,
    RuntimeSnapshot,
    RuntimeStateStore,
    SchedulerSnapshot,
    WorkerSnapshot,
)


def _telemetry(
    *,
    ram: float = 50.0,
    cpu: float = 10.0,
    thermal: ThermalState = ThermalState.COOL,
) -> TelemetrySample:
    return TelemetrySample(
        timestamp=utc_now(),
        cpu_percent=cpu,
        cpu_per_core_percent=(cpu,),
        ram_total_bytes=16 * 1024**3,
        ram_available_bytes=4 * 1024**3,
        ram_used_percent=ram,
        swap_used_percent=12.0,
        disk_read_bytes_per_sec=1024.0,
        disk_write_bytes_per_sec=2048.0,
        thermal_state=thermal,
    )


def _pressured_snapshot() -> RuntimeSnapshot:
    return RuntimeSnapshot(
        telemetry=_telemetry(ram=92.0, cpu=95.0, thermal=ThermalState.HOT),
        system_state=SystemState.YIELDING,
        global_juice_level=18,
        yield_active=True,
        yield_reason="RAM pressure exceeded dynamic yield threshold",
        queue=QueueSnapshot(
            lifecycle_state=QueueLifecycleState.PAUSED,
            queued=5,
            running_foreground=2,
            running_background=1,
            rejected=3,
            completed=12,
        ),
        scheduler=SchedulerSnapshot(
            reserved_foreground_slots=4,
            effective_background_slots=1,
            running_foreground=2,
            running_background=1,
            spare_slots=1,
            immediate_executions=40,
            queued_executions=9,
        ),
        concurrency=ConcurrencyGovernorSnapshot(
            limits=ConcurrencyLimits(
                max_foreground=2,
                max_background=1,
                max_endpoint_foreground=1,
                max_endpoint_background=1,
                reasons=("ram_pressure", "thermal_hot"),
            ),
            endpoints=(
                EndpointCapacitySnapshot(
                    endpoint_name="ollama",
                    max_foreground=1,
                    max_background=1,
                    running_foreground=1,
                    running_background=0,
                ),
            ),
        ),
    )


def _endpoint_record(*, healthy: bool = True) -> EndpointRecord:
    return EndpointRecord(
        name="ollama",
        kind=EndpointKind.OLLAMA,
        base_url="http://127.0.0.1:11434",
        health_path="/api/tags",
        timeout_seconds=30.0,
        capabilities=EndpointCapabilities(
            api_family=EndpointKind.OLLAMA,
            supports_streaming=True,
            supports_model_list=True,
            supports_offload=True,
            default_context_tokens=8192,
            default_output_tokens=2048,
        ),
        health=EndpointHealthStatus(
            healthy=healthy,
            status_code=200 if healthy else 503,
            latency_ms=42.0,
            failure_rate=0.0,
            failure_streak=0,
            last_success_at=utc_now(),
            checked_at=utc_now(),
            detail="ok" if healthy else "service unavailable",
        ),
        routing=EndpointRoutingMetadata(is_default=True),
    )


def _render_text(renderable: object, *, width: int = 140) -> str:
    console = Console(record=True, width=width)
    console.print(renderable)
    return console.export_text()


def _panel_titles(group: Group) -> list[str]:
    titles: list[str] = []
    for renderable in group.renderables:
        title = getattr(renderable, "title", None)
        if title is not None:
            titles.append(str(title))
    return titles


async def test_dashboard_render_structure_has_modular_panels() -> None:
    store = RuntimeStateStore()
    renderable = await Dashboard(store).render()

    assert isinstance(renderable, Group)
    assert len(renderable.renderables) == 8
    titles = _panel_titles(renderable)
    assert titles == [
        "Status",
        "Resources",
        "Policy",
        "Scheduler",
        "Endpoints",
        "Queue Workers",
        "Policy Trace",
        "Recent Decisions",
    ]


async def test_dashboard_empty_state_renders_without_crash() -> None:
    store = RuntimeStateStore()
    output = _render_text(await Dashboard(store).render())

    assert "ComputeCop" in output
    assert "awaiting telemetry" in output
    assert "none configured" in output
    assert "none registered" in output
    assert "no policy trace observed" in output
    assert "no requests observed" in output


async def test_dashboard_pressured_state_renders_all_sections() -> None:
    store = RuntimeStateStore()
    snapshot = _pressured_snapshot()
    await store.update_telemetry(snapshot.telemetry)
    await store.set_policy_state(
        system_state=snapshot.system_state,
        global_juice_level=snapshot.global_juice_level,
        yield_active=snapshot.yield_active,
        yield_reason=snapshot.yield_reason,
    )
    await store.update_queue(snapshot.queue)
    await store.update_scheduler(snapshot.scheduler)
    await store.update_concurrency(snapshot.concurrency)

    output = _render_text(await Dashboard(store).render())

    assert "YIELD" in output
    assert "juice=18" in output
    assert "92.0%" in output
    assert "Scheduler" in output
    assert "paused" in output
    assert "2/1" in output
    assert "yielding" in output
    assert "none configured" in output


async def test_dashboard_panel_renderers_use_stable_dimensions() -> None:
    snapshot = _pressured_snapshot()
    panels = [
        render_header(snapshot),
        render_resource_panel(snapshot),
        render_policy_panel(snapshot),
        render_scheduler_panel(snapshot),
        render_endpoint_panel(snapshot, (_endpoint_record(),)),
        render_worker_panel(snapshot),
        render_trace_panel(snapshot),
        render_decision_panel(snapshot),
    ]

    for panel in panels:
        assert panel.width == DEFAULT_LAYOUT.panel_width


async def test_dashboard_endpoint_panel_renders_registry_records() -> None:
    store = RuntimeStateStore()
    await store.update_concurrency(_pressured_snapshot().concurrency)
    registry = AsyncMock()
    registry.list_records = AsyncMock(return_value=[_endpoint_record()])

    output = _render_text(
        await Dashboard(store, endpoint_registry=registry).render(),
    )

    assert "ollama" in output
    assert "healthy" in output
    assert "42 ms" in output
    assert "1/0" in output
    registry.list_records.assert_awaited_once()


async def test_dashboard_renders_policy_trace_panel() -> None:
    store = RuntimeStateStore()
    trace = PolicyTrace(
        rules=(
            PolicyRuleEvent(
                name="ram_yield",
                status=PolicyRuleStatus.TRIGGERED,
                observed=90.0,
                threshold=85.0,
                penalty=55,
                detail="RAM pressure crossed yield threshold",
            ),
        ),
        summary="RAM pressure crossed yield threshold",
    )
    await store.record_decision(
        AdmissionDecision(
            decision=DecisionType.YIELD,
            request_class=RequestClass.BACKGROUND_REQUEST,
            priority=RequestPriority.BACKGROUND,
            budget=JuiceBudget(
                juice_level=15,
                max_context_tokens=1024,
                max_output_tokens=256,
                concurrency_limit=1,
                reason="RAM pressure",
            ),
            reason="RAM pressure",
            correlation_id="dashboard-trace",
            trace=trace,
        )
    )

    output = _render_text(await Dashboard(store).render())
    assert "Policy Trace" in output
    assert "ram_yield" in output
    assert "RAM pressure crossed yield threshold" in output


async def test_dashboard_renders_worker_state_panel() -> None:
    store = RuntimeStateStore()
    await store.update_queue(
        QueueSnapshot(
            lifecycle_state=QueueLifecycleState.DRAINING,
            queued=2,
            running_background=1,
            workers=(
                WorkerSnapshot(
                    worker_id="computecop-queue-worker-0",
                    state=WorkerState.RUNNING,
                    active_correlation_id="worker-correlation-id",
                ),
                WorkerSnapshot(
                    worker_id="computecop-queue-worker-1",
                    state=WorkerState.IDLE,
                ),
            ),
        )
    )

    output = _render_text(await Dashboard(store).render())
    assert "Queue Workers" in output
    assert "computecop-queue-worker-0" in output
    assert "running" in output
    assert "worker-correlation-id" in output
    assert "draining" in output


async def test_dashboard_shows_persistence_warning_when_disabled() -> None:
    store = RuntimeStateStore()
    await store.set_event_persistence(
        enabled=False, disabled_reason="Permission denied: /root/events.jsonl"
    )

    renderable = await Dashboard(store).render()
    assert isinstance(renderable, Group)
    assert len(renderable.renderables) == 9

    output = _render_text(renderable)
    assert "Event persistence disabled" in output
    assert "Permission denied" in output


async def test_dashboard_hides_persistence_warning_when_healthy() -> None:
    store = RuntimeStateStore()
    await store.set_event_persistence(enabled=True, disabled_reason=None)

    output = _render_text(await Dashboard(store).render())
    assert "Event persistence disabled" not in output


def test_cli_dashboard_wires_runtime_dependencies() -> None:
    from pathlib import Path

    source = Path("src/computecop/cli.py").read_text(encoding="utf-8")
    assert "DashboardQueueController" in source
    assert "endpoint_registry=runtime.endpoint_registry" in source
    assert "queue_controller=queue_controller" in source
