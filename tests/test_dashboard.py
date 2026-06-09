from __future__ import annotations

from rich.console import Console

from computecop.dashboard import Dashboard
from computecop.models import (
    AdmissionDecision,
    DecisionType,
    JuiceBudget,
    PolicyRuleEvent,
    PolicyRuleStatus,
    PolicyTrace,
    QueueLifecycleState,
    RequestClass,
    RequestPriority,
    WorkerState,
)
from computecop.state import (
    QueueSnapshot,
    RuntimeStateStore,
    WorkerSnapshot,
)


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

    renderable = await Dashboard(store).render()
    console = Console(record=True, width=140)
    console.print(renderable)
    output = console.export_text()
    assert "Why" in output
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

    renderable = await Dashboard(store).render()
    console = Console(record=True, width=140)
    console.print(renderable)
    output = console.export_text()
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
    console = Console(record=True, width=140)
    console.print(renderable)
    output = console.export_text()
    assert "Event persistence disabled" in output
    assert "Permission denied" in output


async def test_dashboard_hides_persistence_warning_when_healthy() -> None:
    store = RuntimeStateStore()
    await store.set_event_persistence(enabled=True, disabled_reason=None)

    renderable = await Dashboard(store).render()
    console = Console(record=True, width=140)
    console.print(renderable)
    output = console.export_text()
    assert "Event persistence disabled" not in output
