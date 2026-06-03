from __future__ import annotations

import asyncio

import pytest

from computecop.config import QueueConfig
from computecop.models import RequestClass, RequestMetadata, RequestPriority, SystemState
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


async def _answer(value: str) -> str:
    return value
