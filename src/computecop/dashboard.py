"""Rich terminal dashboard for ComputeCop."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live

from computecop.dashboard_controls import (
    DashboardInteractionState,
    DashboardKeyHandler,
    DashboardQueueController,
)
from computecop.dashboard_input import DashboardKeyReader
from computecop.dashboard_panels import (
    DEFAULT_LAYOUT,
    DashboardLayout,
    render_decision_panel,
    render_endpoint_panel,
    render_footer,
    render_header,
    render_persistence_warning,
    render_policy_panel,
    render_resource_panel,
    render_scheduler_panel,
    render_trace_panel,
    render_worker_panel,
)
from computecop.endpoints import EndpointRecord
from computecop.state import RuntimeStateStore

if TYPE_CHECKING:
    from computecop.endpoints import EndpointCapabilityRegistry


class Dashboard:
    """Render a modular live terminal dashboard."""

    def __init__(
        self,
        state: RuntimeStateStore,
        refresh_seconds: float = 0.5,
        *,
        endpoint_registry: EndpointCapabilityRegistry | None = None,
        queue_controller: DashboardQueueController | None = None,
        layout: DashboardLayout = DEFAULT_LAYOUT,
        interactive: bool | None = None,
    ) -> None:
        self.state = state
        self.refresh_seconds = refresh_seconds
        self._endpoint_registry = endpoint_registry
        self._queue_controller = queue_controller
        self._layout = layout
        if interactive is None:
            self._interactive = queue_controller is not None
        else:
            self._interactive = interactive

    async def run(self) -> None:
        """Run until cancelled or the operator presses quit."""

        interaction = DashboardInteractionState()
        key_reader = DashboardKeyReader()
        key_handler = (
            DashboardKeyHandler(self._queue_controller)
            if self._queue_controller is not None
            else None
        )
        if self._interactive and key_handler is not None:
            key_reader.start()
        try:
            with Live(
                await self.render(interaction),
                refresh_per_second=4,
                screen=False,
            ) as live:
                while True:
                    if key_handler is not None:
                        for key in key_reader.poll_keys():
                            await key_handler.handle(key, interaction)
                            if key_handler.quit_requested:
                                raise asyncio.CancelledError
                        key_handler.expire_timers(interaction)
                    await asyncio.sleep(self.refresh_seconds)
                    live.update(await self.render(interaction))
        finally:
            key_reader.stop()

    async def render(
        self,
        interaction: DashboardInteractionState | None = None,
    ) -> Group:
        view = interaction or DashboardInteractionState()
        snapshot = await self.state.snapshot()
        endpoints = await self._load_endpoint_records()
        detail_mode = view.detail_mode
        panels = [
            render_header(snapshot, layout=self._layout),
            render_resource_panel(snapshot, layout=self._layout, detail_mode=detail_mode),
            render_policy_panel(snapshot, layout=self._layout),
            render_scheduler_panel(snapshot, layout=self._layout),
            render_endpoint_panel(snapshot, endpoints, layout=self._layout),
            render_worker_panel(snapshot, layout=self._layout),
            render_trace_panel(snapshot, layout=self._layout, detail_mode=detail_mode),
            render_decision_panel(snapshot, layout=self._layout, detail_mode=detail_mode),
        ]
        warning = render_persistence_warning(snapshot, layout=self._layout)
        if warning is not None:
            panels.insert(1, warning)
        if self._interactive:
            draining = (
                self._queue_controller.draining if self._queue_controller is not None else False
            )
            panels.append(
                render_footer(
                    detail_mode=detail_mode,
                    pending_action=view.pending_action,
                    status_message=view.status_message,
                    draining=draining,
                    layout=self._layout,
                )
            )
        return Group(*panels)

    async def _load_endpoint_records(self) -> tuple[EndpointRecord, ...]:
        if self._endpoint_registry is None:
            return ()
        records = await self._endpoint_registry.list_records()
        return tuple(records)
