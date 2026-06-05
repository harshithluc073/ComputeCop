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
    RequestClass,
    RequestPriority,
)
from computecop.state import RuntimeStateStore


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
