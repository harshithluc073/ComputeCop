from __future__ import annotations

import asyncio
import contextlib
from time import monotonic

import pytest

from computecop.app import build_runtime
from computecop.config import QueueConfig, RuntimeConfig
from computecop.models import (
    AdmissionDecision,
    DecisionType,
    JuiceBudget,
    QueueLifecycleState,
    RequestClass,
    RequestMetadata,
    RequestPriority,
    SystemState,
    WorkerState,
)
from computecop.request_queue import AsyncRequestQueue, QueueFullError
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
    worker = asyncio.create_task(queue.run_worker("worker-0"))
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
    worker = asyncio.create_task(queue.run_worker("worker-0"))
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


def _background_metadata() -> RequestMetadata:
    return RequestMetadata(
        method="POST",
        path="/api/chat",
        headers={},
        request_class=RequestClass.BACKGROUND_REQUEST,
        priority=RequestPriority.BACKGROUND,
    )


@pytest.mark.asyncio
async def test_queue_pause_rejects_background_submit() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    await queue.pause()
    with pytest.raises(QueueFullError, match="not accepting background work"):
        await queue.submit(_background_metadata(), lambda: _answer("blocked"))
    assert queue.snapshot().lifecycle_state == QueueLifecycleState.PAUSED


@pytest.mark.asyncio
async def test_queue_drain_rejects_new_background_work() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    release = asyncio.Event()

    async def slow_runner() -> str:
        await release.wait()
        return "done"

    worker = asyncio.create_task(queue.run_worker("worker-0"))
    submit_task = asyncio.create_task(queue.submit(_background_metadata(), slow_runner))
    await asyncio.sleep(0)
    await queue.drain(deadline=monotonic() + 5)
    with pytest.raises(QueueFullError, match="not accepting background work"):
        await queue.submit(_background_metadata(), lambda: _answer("blocked"))
    release.set()
    try:
        assert await submit_task == "done"
        drained = await queue.drain(deadline=monotonic() + 1)
        assert drained is True
        assert queue.snapshot().lifecycle_state == QueueLifecycleState.DRAINING
    finally:
        await queue.close()
        worker.cancel()


@pytest.mark.asyncio
async def test_queue_drain_waits_for_queued_background_work() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    gate = asyncio.Event()

    async def gated_runner() -> str:
        await gate.wait()
        return "finished"

    worker = asyncio.create_task(queue.run_worker("worker-0"))
    task = asyncio.create_task(queue.submit(_background_metadata(), gated_runner))
    await asyncio.sleep(0)
    assert queue.snapshot().queued == 1
    drain_task = asyncio.create_task(queue.drain(deadline=monotonic() + 2))
    await asyncio.sleep(0.05)
    assert not drain_task.done()
    gate.set()
    try:
        assert await task == "finished"
        assert await drain_task is True
        assert queue.snapshot().running_background == 0
    finally:
        await queue.close()
        worker.cancel()


@pytest.mark.asyncio
async def test_queue_close_cancels_pending_background_work() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    worker = asyncio.create_task(queue.run_worker("worker-0"))
    never = asyncio.Event()

    async def wait_forever() -> str:
        await never.wait()
        return "never"

    submit_task = asyncio.create_task(queue.submit(_background_metadata(), wait_forever))
    await asyncio.sleep(0)
    await queue.close()
    with pytest.raises(asyncio.CancelledError):
        await submit_task
    worker.cancel()
    snapshot = queue.snapshot()
    assert snapshot.lifecycle_state == QueueLifecycleState.CLOSED
    assert snapshot.queued == 0


@pytest.mark.asyncio
async def test_runtime_registers_workers_at_start() -> None:
    runtime = build_runtime(RuntimeConfig(policy={"max_background_concurrency": 2}))
    await runtime.start()
    try:
        workers = runtime.queue.snapshot().workers
        assert len(workers) == 2
        assert {worker.worker_id for worker in workers} == {
            "computecop-queue-worker-0",
            "computecop-queue-worker-1",
        }
        assert all(worker.state == WorkerState.IDLE for worker in workers)
    finally:
        await runtime.stop(drain_timeout_seconds=0)


@pytest.mark.asyncio
async def test_runtime_stop_drains_then_closes_queue() -> None:
    runtime = build_runtime(RuntimeConfig(queue=QueueConfig(shutdown_drain_seconds=0.0)))
    await runtime.start()
    gate = asyncio.Event()

    async def gated_runner() -> str:
        await gate.wait()
        return "ok"

    submit_task = asyncio.create_task(runtime.queue.submit(_background_metadata(), gated_runner))
    await asyncio.sleep(0)
    gate.set()
    await submit_task
    await runtime.stop(drain_timeout_seconds=0)
    await runtime.stop(drain_timeout_seconds=0)
    snapshot = runtime.queue.snapshot()
    assert snapshot.lifecycle_state == QueueLifecycleState.CLOSED
    await runtime.upstream.close()


@pytest.mark.asyncio
async def test_queue_worker_states_are_tracked() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    release = asyncio.Event()
    metadata = _background_metadata()

    async def runner() -> str:
        await release.wait()
        return "done"

    worker = asyncio.create_task(queue.run_worker("worker-a"))
    submit_task = asyncio.create_task(queue.submit(metadata, runner))
    running = None
    for _ in range(100):
        await asyncio.sleep(0.01)
        running = next(
            (
                worker_snapshot
                for worker_snapshot in queue.snapshot().workers
                if worker_snapshot.worker_id == "worker-a"
            ),
            None,
        )
        if running is not None and running.state == WorkerState.RUNNING:
            break
    assert running is not None
    assert running.state == WorkerState.RUNNING
    assert running.active_correlation_id == metadata.correlation_id
    release.set()
    try:
        assert await submit_task == "done"
    finally:
        await queue.close()
        worker.cancel()
    stopped = next(
        worker_snapshot
        for worker_snapshot in queue.snapshot().workers
        if worker_snapshot.worker_id == "worker-a"
    )
    assert stopped.state in {WorkerState.IDLE, WorkerState.STOPPED, WorkerState.STOPPING}


async def _answer(value: str) -> str:
    return value


@pytest.mark.asyncio
async def test_queue_inspect() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    metadata = _background_metadata()
    metadata = RequestMetadata(
        method=metadata.method,
        path=metadata.path,
        headers=metadata.headers,
        request_class=metadata.request_class,
        priority=metadata.priority,
        correlation_id="test-corr-id",
        client_host=metadata.client_host,
        model="test-model",
        endpoint_name="test-endpoint",
        received_at=metadata.received_at,
    )

    task = asyncio.create_task(queue.submit(metadata, lambda: _answer("done")))
    await asyncio.sleep(0.01)

    items = await queue.inspect()
    assert len(items) == 1
    assert items[0]["correlation_id"] == "test-corr-id"
    assert items[0]["class"] == RequestClass.BACKGROUND_REQUEST.value
    assert items[0]["priority"] == RequestPriority.BACKGROUND.value
    assert items[0]["endpoint"] == "test-endpoint"
    assert items[0]["estimated_tokens"] == 0
    assert items[0]["age"] >= 0

    await queue.close()
    with contextlib.suppress(BaseException):
        await task

