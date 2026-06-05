from __future__ import annotations

import asyncio

import pytest

from computecop.config import QueueConfig
from computecop.models import (
    AdmissionDecision,
    DecisionType,
    JuiceBudget,
    RequestClass,
    RequestMetadata,
    RequestPriority,
    SystemState,
)
from computecop.request_queue import AsyncRequestQueue
from computecop.state import RuntimeStateStore


@pytest.mark.asyncio
async def test_state_policy_snapshot_updates() -> None:
    store = RuntimeStateStore()
    await store.set_policy_state(
        system_state=SystemState.PRESSURED,
        global_juice_level=42,
        yield_active=False,
        yield_reason=None,
    )
    snapshot = await store.snapshot()
    assert snapshot.system_state == SystemState.PRESSURED
    assert snapshot.global_juice_level == 42


@pytest.mark.asyncio
async def test_state_decision_lookup_by_correlation_id() -> None:
    store = RuntimeStateStore()
    decision = AdmissionDecision(
        decision=DecisionType.ALLOW,
        request_class=RequestClass.USER_PROMPT,
        priority=RequestPriority.FOREGROUND,
        budget=JuiceBudget(
            juice_level=100,
            max_context_tokens=8192,
            max_output_tokens=2048,
            concurrency_limit=1,
            reason="test",
        ),
        reason="test decision",
        correlation_id="lookup-id",
    )
    await store.record_decision(decision)
    assert await store.decision_for_correlation_id("lookup-id") == decision
    assert await store.decision_for_correlation_id("missing") is None


@pytest.mark.asyncio
async def test_queue_executes_submitted_work() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    worker = asyncio.create_task(queue.run_worker())
    metadata = RequestMetadata(
        method="POST",
        path="/api/chat",
        headers={},
        request_class=RequestClass.BACKGROUND_REQUEST,
        priority=RequestPriority.BACKGROUND,
    )
    try:
        result = await queue.submit(metadata, lambda: _answer("ok"))
        assert result == "ok"
        assert queue.counters().completed == 1
    finally:
        await queue.close()
        worker.cancel()


@pytest.mark.asyncio
async def test_queue_change_callback_reports_counters() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    observed = []
    release = asyncio.Event()

    async def on_change(counters):
        observed.append(counters)

    async def runner() -> str:
        await release.wait()
        return "done"

    queue.set_change_callback(on_change)
    worker = asyncio.create_task(queue.run_worker())
    metadata = RequestMetadata(
        method="POST",
        path="/api/chat",
        headers={},
        request_class=RequestClass.BACKGROUND_REQUEST,
        priority=RequestPriority.BACKGROUND,
    )
    task = asyncio.create_task(queue.submit(metadata, runner))
    await asyncio.sleep(0)
    release.set()
    try:
        assert await task == "done"
    finally:
        await queue.close()
        worker.cancel()

    assert any(counters.queued == 1 for counters in observed)
    assert observed[-1].completed == 1


async def _answer(value: str) -> str:
    return value
