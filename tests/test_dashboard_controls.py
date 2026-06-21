from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from computecop.config import QueueConfig
from computecop.dashboard import Dashboard
from computecop.dashboard_controls import (
    DashboardInteractionState,
    DashboardKeyHandler,
    DashboardQueueController,
)
from computecop.dashboard_input import DashboardKeyReader
from computecop.dashboard_panels import render_footer
from computecop.models import QueueLifecycleState
from computecop.request_queue import AsyncRequestQueue
from computecop.state import RuntimeStateStore


@pytest.mark.asyncio
async def test_key_handler_pause_and_resume_are_instant() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    controller = DashboardQueueController(queue, drain_seconds=5.0)
    handler = DashboardKeyHandler(controller)
    state = DashboardInteractionState()

    await handler.handle("p", state)
    assert queue.lifecycle_state == QueueLifecycleState.PAUSED
    assert state.status_message == "Queue paused"
    assert state.pending_action is None

    await handler.handle("r", state)
    assert queue.lifecycle_state == QueueLifecycleState.ACCEPTING
    assert state.status_message == "Queue resumed"


@pytest.mark.asyncio
async def test_key_handler_drain_requires_confirmation() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    controller = DashboardQueueController(queue, drain_seconds=5.0)
    handler = DashboardKeyHandler(controller)
    state = DashboardInteractionState()

    await handler.handle("d", state)
    assert state.pending_action == "drain"
    assert queue.lifecycle_state == QueueLifecycleState.ACCEPTING
    assert "confirm" in (state.status_message or "").lower()

    await handler.handle("d", state)
    assert state.pending_action is None
    assert queue.lifecycle_state == QueueLifecycleState.DRAINING
    assert state.status_message == "Queue drain started"


@pytest.mark.asyncio
async def test_key_handler_cancel_drain_confirmation() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    controller = DashboardQueueController(queue, drain_seconds=5.0)
    handler = DashboardKeyHandler(controller)
    state = DashboardInteractionState()

    await handler.handle("d", state)
    await handler.handle("c", state)

    assert state.pending_action is None
    assert queue.lifecycle_state == QueueLifecycleState.ACCEPTING
    assert state.status_message == "Drain cancelled"


@pytest.mark.asyncio
async def test_key_handler_confirmation_expires() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    controller = DashboardQueueController(queue, drain_seconds=5.0)
    handler = DashboardKeyHandler(controller, confirmation_timeout=0.01)
    state = DashboardInteractionState()

    await handler.handle("d", state)
    await asyncio.sleep(0.02)
    handler.expire_timers(state)

    assert state.pending_action is None
    assert state.status_message == "Confirmation expired"


@pytest.mark.asyncio
async def test_key_handler_toggles_detail_mode() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    handler = DashboardKeyHandler(DashboardQueueController(queue, drain_seconds=5.0))
    state = DashboardInteractionState()

    await handler.handle("t", state)
    assert state.detail_mode is True
    assert state.status_message == "Detail view enabled"

    await handler.handle("t", state)
    assert state.detail_mode is False
    assert state.status_message == "Detail view disabled"


@pytest.mark.asyncio
async def test_key_handler_quit_sets_flag() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    handler = DashboardKeyHandler(DashboardQueueController(queue, drain_seconds=5.0))
    state = DashboardInteractionState()

    await handler.handle("q", state)
    assert handler.quit_requested is True


def test_footer_renders_shortcuts_and_confirmation_prompt() -> None:
    from rich.console import Console

    footer = render_footer(
        detail_mode=True,
        pending_action="drain",
        status_message="Press D again to confirm drain (C to cancel)",
        draining=False,
    )
    console = Console(record=True, width=140)
    console.print(footer)
    output = console.export_text()

    assert "Controls" in output
    assert "[P]ause" in output
    assert "[R]esume" in output
    assert "[D]rain" in output
    assert "[T]oggle detail" in output
    assert "[Q]uit" in output
    assert "Confirm: [D]" in output
    assert "Detail: ON" in output
    assert "confirm drain" in output


@pytest.mark.asyncio
async def test_interactive_dashboard_render_includes_footer() -> None:
    store = RuntimeStateStore()
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    controller = DashboardQueueController(queue, drain_seconds=5.0)
    renderable = await Dashboard(
        store,
        queue_controller=controller,
        interactive=True,
    ).render()

    from rich.console import Group

    assert isinstance(renderable, Group)
    titles = [str(getattr(panel, "title", "")) for panel in renderable.renderables]
    assert "Controls" in titles


@pytest.mark.asyncio
async def test_dashboard_run_exits_on_quit_key() -> None:
    store = RuntimeStateStore()
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    controller = DashboardQueueController(queue, drain_seconds=5.0)
    dashboard = Dashboard(
        store,
        refresh_seconds=0.01,
        queue_controller=controller,
        interactive=True,
    )

    with patch.object(DashboardKeyReader, "poll_keys", side_effect=[["q"], []]):
        with pytest.raises(asyncio.CancelledError):
            await dashboard.run()


def test_key_reader_poll_keys_returns_enqueued_keys() -> None:
    reader = DashboardKeyReader()
    reader._enqueue("p")
    reader._enqueue("r")
    assert reader.poll_keys() == ["p", "r"]
    assert reader.poll_keys() == []


@pytest.mark.asyncio
async def test_queue_controller_start_drain_transitions_queue() -> None:
    queue = AsyncRequestQueue(QueueConfig(max_size=4))
    controller = DashboardQueueController(queue, drain_seconds=5.0)

    await controller.start_drain()
    assert queue.lifecycle_state == QueueLifecycleState.DRAINING
