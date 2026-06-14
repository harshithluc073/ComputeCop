from __future__ import annotations

import asyncio
from dataclasses import replace
from time import monotonic

import pytest

from computecop.config import PolicyConfig, QueueConfig
from computecop.config import PolicyConfig as SchedulerPolicyConfig
from computecop.models import (
    PolicyRuleEvent,
    PolicyRuleStatus,
    PolicyTrace,
    RequestClass,
    RequestMetadata,
    RequestPriority,
    SystemState,
)
from computecop.policy import ConcurrencyLimits, PressureReport, compute_concurrency_limits
from computecop.request_queue import AsyncRequestQueue
from computecop.scheduler import (
    AdaptiveScheduler,
    build_scheduled_work,
    effective_background_slots,
    estimate_work_cost,
)


def _metadata(
    *,
    priority: RequestPriority = RequestPriority.BACKGROUND,
    request_class: RequestClass = RequestClass.BACKGROUND_REQUEST,
    correlation_id: str = "test-id",
    model: str | None = "llama3.1",
) -> RequestMetadata:
    return RequestMetadata(
        method="POST",
        path="/api/chat",
        headers={},
        request_class=request_class,
        priority=priority,
        correlation_id=correlation_id,
        model=model,
    )


def _pressure_report(
    *,
    system_state: SystemState = SystemState.NORMAL,
    yield_active: bool = False,
) -> PressureReport:
    trace = PolicyTrace(
        rules=(
            PolicyRuleEvent(
                name="test",
                status=PolicyRuleStatus.OBSERVED,
                observed=10.0,
                threshold=80.0,
                penalty=0,
                detail="test pressure",
            ),
        ),
        system_state=system_state,
        summary="test pressure",
    )
    report = PressureReport(
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
        concurrency_limits=ConcurrencyLimits(
            max_foreground=4,
            max_background=4,
            max_endpoint_foreground=2,
            max_endpoint_background=1,
            reasons=("test",),
        ),
    )
    limits = compute_concurrency_limits(
        report,
        SchedulerPolicyConfig(max_background_concurrency=4),
    )
    return replace(report, concurrency_limits=limits)


def test_estimate_work_cost_prefers_foreground() -> None:
    foreground = estimate_work_cost(
        _metadata(priority=RequestPriority.FOREGROUND, request_class=RequestClass.USER_PROMPT)
    )
    background = estimate_work_cost(_metadata(priority=RequestPriority.BULK))
    assert foreground > background


def test_build_scheduled_work_captures_metadata() -> None:
    metadata = RequestMetadata(
        method="POST",
        path="/api/chat",
        headers={},
        request_class=RequestClass.BACKGROUND_REQUEST,
        priority=RequestPriority.BACKGROUND,
        correlation_id="scheduled-work",
        model="llama3.1",
        endpoint_name="ollama",
    )
    work = build_scheduled_work(metadata, lambda: _answer("ok"), deadline=monotonic() + 30)
    assert work.endpoint_name == "ollama"
    assert work.model == "llama3.1"
    assert work.estimated_cost > 0
    assert work.request_class == RequestClass.BACKGROUND_REQUEST


def test_effective_background_slots_shrink_under_pressure() -> None:
    policy = PolicyConfig(max_background_concurrency=4)
    assert effective_background_slots(_pressure_report(), policy) == 4
    assert (
        effective_background_slots(_pressure_report(system_state=SystemState.RECOVERING), policy)
        == 3
    )
    assert (
        effective_background_slots(_pressure_report(system_state=SystemState.PRESSURED), policy)
        == 2
    )
    assert effective_background_slots(_pressure_report(yield_active=True), policy) == 0


@pytest.mark.asyncio
async def test_scheduler_snapshot_tracks_capacity() -> None:
    scheduler = _scheduler(
        policy=PolicyConfig(max_foreground_concurrency=2, max_background_concurrency=2)
    )
    snapshot = scheduler.snapshot()
    assert snapshot.reserved_foreground_slots == 2
    assert snapshot.max_background_slots == 2
    assert snapshot.total_capacity == 4


@pytest.mark.asyncio
async def test_scheduler_update_pressure_changes_background_slots() -> None:
    scheduler = _scheduler(policy=PolicyConfig(max_background_concurrency=4))
    await scheduler.update_pressure(_pressure_report(system_state=SystemState.PRESSURED))
    assert scheduler.snapshot().effective_background_slots == 2
    await scheduler.update_pressure(_pressure_report(yield_active=True))
    assert scheduler.snapshot().effective_background_slots == 0


@pytest.mark.asyncio
async def test_scheduler_foreground_executes_without_waiting_for_background() -> None:
    scheduler = _scheduler(
        policy=PolicyConfig(max_foreground_concurrency=1, max_background_concurrency=1)
    )
    gate = asyncio.Event()

    async def hold_background() -> str:
        await gate.wait()
        return "background"

    await scheduler.start()
    queue = scheduler.queue
    background_task = asyncio.create_task(
        scheduler.execute_queued(_metadata(correlation_id="bg"), hold_background)
    )
    await asyncio.sleep(0.05)

    result = await scheduler.execute_immediate(
        _metadata(
            priority=RequestPriority.FOREGROUND,
            request_class=RequestClass.USER_PROMPT,
            correlation_id="fg",
        ),
        lambda: _answer("foreground"),
    )
    assert result == "foreground"
    gate.set()
    try:
        assert await background_task == "background"
    finally:
        await queue.close()
        await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_blocks_background_when_effective_slots_are_zero() -> None:
    scheduler = _scheduler(
        policy=PolicyConfig(max_foreground_concurrency=1, max_background_concurrency=1)
    )
    await scheduler.update_pressure(_pressure_report(yield_active=True))
    acquired = asyncio.Event()

    async def runner() -> str:
        acquired.set()
        return "should-not-run"

    task = asyncio.create_task(
        scheduler.execute_immediate(_metadata(correlation_id="blocked"), runner)
    )
    await asyncio.sleep(0.1)
    assert not acquired.is_set()
    await scheduler.update_pressure(_pressure_report())
    result = await asyncio.wait_for(task, timeout=2)
    assert result == "should-not-run"


@pytest.mark.asyncio
async def test_queue_aging_promotes_long_waiting_bulk_work() -> None:
    queue = AsyncRequestQueue(
        QueueConfig(max_size=8, aging_interval_seconds=0.1, default_timeout_seconds=30.0)
    )
    order: list[str] = []

    async def record(label: str) -> str:
        order.append(label)
        return label

    worker = asyncio.create_task(queue.run_worker("worker-0"))
    bulk = _metadata(priority=RequestPriority.BULK, correlation_id="bulk")
    interactive = _metadata(priority=RequestPriority.INTERACTIVE, correlation_id="interactive")

    first = asyncio.create_task(queue.submit(bulk, lambda: record("bulk")))
    await asyncio.sleep(0)
    second = asyncio.create_task(queue.submit(interactive, lambda: record("interactive")))
    await asyncio.sleep(0.25)
    try:
        await asyncio.gather(first, second)
    finally:
        await queue.close()
        worker.cancel()

    assert order[0] == "bulk"


@pytest.mark.asyncio
async def test_scheduler_execute_queued_uses_capacity_hooks() -> None:
    scheduler = _scheduler(
        policy=PolicyConfig(max_foreground_concurrency=2, max_background_concurrency=1)
    )
    await scheduler.start()
    try:
        result = await scheduler.execute_queued(
            _metadata(correlation_id="queued"),
            lambda: _answer("queued-result"),
        )
        assert result == "queued-result"
        snapshot = scheduler.snapshot()
        assert snapshot.queued_executions == 0
        assert scheduler.queue.snapshot().completed == 1
    finally:
        await scheduler.queue.close()
        await scheduler.stop()


@pytest.mark.asyncio
async def test_runtime_state_includes_scheduler_snapshot() -> None:
    from computecop.app import build_runtime
    from computecop.config import RuntimeConfig
    from computecop.state import SchedulerSnapshot

    runtime = build_runtime(
        RuntimeConfig(policy={"max_foreground_concurrency": 3, "max_background_concurrency": 2})
    )
    await runtime.state.update_scheduler(
        SchedulerSnapshot(
            reserved_foreground_slots=3,
            max_background_slots=2,
            effective_background_slots=2,
            total_capacity=5,
        )
    )
    snapshot = await runtime.state.snapshot()
    assert snapshot.scheduler.reserved_foreground_slots == 3
    assert snapshot.scheduler.total_capacity == 5
    await runtime.upstream.close()


def _scheduler(*, policy: PolicyConfig | None = None) -> AdaptiveScheduler:
    policy_config = policy or PolicyConfig()
    queue_config = QueueConfig(max_size=8, aging_interval_seconds=0.1)
    queue = AsyncRequestQueue(queue_config)
    return AdaptiveScheduler(queue, policy_config=policy_config, queue_config=queue_config)


async def _answer(value: str) -> str:
    return value


@pytest.mark.asyncio
async def test_scheduler_stress_many_requests() -> None:
    queue_config = QueueConfig(max_size=100, aging_interval_seconds=0.1)
    policy_config = PolicyConfig(max_foreground_concurrency=4, max_background_concurrency=4)
    queue = AsyncRequestQueue(queue_config)
    scheduler = AdaptiveScheduler(queue, policy_config=policy_config, queue_config=queue_config)

    await scheduler.start()

    active_bg_runs = 0
    max_concurrent_bg = 0
    bg_lock = asyncio.Lock()

    async def runner_bg(idx: int) -> int:
        nonlocal active_bg_runs, max_concurrent_bg
        async with bg_lock:
            active_bg_runs += 1
            if active_bg_runs > max_concurrent_bg:
                max_concurrent_bg = active_bg_runs
        await asyncio.sleep(0.01)
        async with bg_lock:
            active_bg_runs -= 1
        return idx

    async def runner_fg(idx: int) -> int:
        await asyncio.sleep(0.01)
        return idx

    try:
        bg_tasks = [
            scheduler.execute_queued(
                _metadata(correlation_id=f"bg-{i}", priority=RequestPriority.BACKGROUND),
                lambda i=i: runner_bg(i),
            )
            for i in range(50)
        ]

        fg_tasks = [
            scheduler.execute_immediate(
                _metadata(
                    priority=RequestPriority.FOREGROUND,
                    request_class=RequestClass.USER_PROMPT,
                    correlation_id=f"fg-{i}",
                ),
                lambda i=i: runner_fg(i + 100),
            )
            for i in range(10)
        ]

        results = await asyncio.gather(*(bg_tasks + fg_tasks))

        assert len(results) == 60
        assert max_concurrent_bg <= 4

        snapshot = scheduler.snapshot()
        assert snapshot.running_foreground == 0
        assert snapshot.running_background == 0
        assert snapshot.queued_executions == 0
        assert snapshot.immediate_executions == 0

    finally:
        await scheduler.queue.close()
        await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_mock_failures_and_recoveries() -> None:
    queue_config = QueueConfig(max_size=10, aging_interval_seconds=0.1)
    policy_config = PolicyConfig(max_foreground_concurrency=2, max_background_concurrency=2)
    queue = AsyncRequestQueue(queue_config)
    scheduler = AdaptiveScheduler(queue, policy_config=policy_config, queue_config=queue_config)

    await scheduler.start()

    async def failing_runner() -> str:
        await asyncio.sleep(0.01)
        raise ValueError("simulated endpoint failure")

    async def succeeding_runner() -> str:
        await asyncio.sleep(0.01)
        return "success"

    try:
        with pytest.raises(ValueError, match="simulated endpoint failure"):
            await scheduler.execute_queued(
                _metadata(correlation_id="fail-1"),
                failing_runner,
            )

        snapshot_after_failure = scheduler.snapshot()
        assert snapshot_after_failure.running_background == 0
        assert snapshot_after_failure.queued_executions == 0

        res = await scheduler.execute_queued(
            _metadata(correlation_id="success-1"),
            succeeding_runner,
        )
        assert res == "success"

        snapshot_after_success = scheduler.snapshot()
        assert snapshot_after_success.running_background == 0
        assert snapshot_after_success.queued_executions == 0

    finally:
        await scheduler.queue.close()
        await scheduler.stop()


@pytest.mark.asyncio
async def test_scheduler_no_counter_drift_after_cancellation() -> None:
    queue_config = QueueConfig(max_size=10, aging_interval_seconds=0.1)
    policy_config = PolicyConfig(max_foreground_concurrency=1, max_background_concurrency=1)
    queue = AsyncRequestQueue(queue_config)
    scheduler = AdaptiveScheduler(queue, policy_config=policy_config, queue_config=queue_config)

    await scheduler.start()

    start_event = asyncio.Event()
    finish_event = asyncio.Event()

    async def slow_runner() -> str:
        start_event.set()
        try:
            await finish_event.wait()
            return "done"
        except asyncio.CancelledError:
            raise

    async def queued_runner() -> str:
        return "queued-done"

    try:
        first_task = asyncio.create_task(
            scheduler.execute_queued(_metadata(correlation_id="slow"), slow_runner)
        )
        await start_event.wait()

        second_task = asyncio.create_task(
            scheduler.execute_queued(_metadata(correlation_id="queued"), queued_runner)
        )
        await asyncio.sleep(0.05)

        snapshot_before = scheduler.snapshot()
        assert snapshot_before.running_background == 1
        assert snapshot_before.queued_executions == 2

        second_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second_task

        snapshot_after_cancel_queued = scheduler.snapshot()
        assert snapshot_after_cancel_queued.running_background == 1
        assert snapshot_after_cancel_queued.queued_executions == 1

        first_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_task

        snapshot_while_blocked = scheduler.snapshot()
        assert snapshot_while_blocked.running_background == 1

        finish_event.set()
        await asyncio.sleep(0.05)

        snapshot_after_finish = scheduler.snapshot()
        assert snapshot_after_finish.running_background == 0
        assert snapshot_after_finish.queued_executions == 0

        res = await scheduler.execute_queued(_metadata(correlation_id="test-new"), queued_runner)
        assert res == "queued-done"

        snapshot_final = scheduler.snapshot()
        assert snapshot_final.running_background == 0
        assert snapshot_final.queued_executions == 0

    finally:
        await scheduler.queue.close()
        await scheduler.stop()


