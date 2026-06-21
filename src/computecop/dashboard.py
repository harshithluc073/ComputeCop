"""Rich terminal dashboard for ComputeCop."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live

from computecop.dashboard_panels import (
    DEFAULT_LAYOUT,
    DashboardLayout,
    render_decision_panel,
    render_endpoint_panel,
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
        layout: DashboardLayout = DEFAULT_LAYOUT,
    ) -> None:
        self.state = state
        self.refresh_seconds = refresh_seconds
        self._endpoint_registry = endpoint_registry
        self._layout = layout

    async def run(self) -> None:
        """Run until cancelled."""

        with Live(await self.render(), refresh_per_second=4, screen=False) as live:
            while True:
                await asyncio.sleep(self.refresh_seconds)
                live.update(await self.render())

    async def render(self) -> Group:
        snapshot = await self.state.snapshot()
        endpoints = await self._load_endpoint_records()
        panels = [
            render_header(snapshot, layout=self._layout),
            render_resource_panel(snapshot, layout=self._layout),
            render_policy_panel(snapshot, layout=self._layout),
            render_scheduler_panel(snapshot, layout=self._layout),
            render_endpoint_panel(snapshot, endpoints, layout=self._layout),
            render_worker_panel(snapshot, layout=self._layout),
            render_trace_panel(snapshot, layout=self._layout),
            render_decision_panel(snapshot, layout=self._layout),
        ]
        warning = render_persistence_warning(snapshot, layout=self._layout)
        if warning is not None:
            panels.insert(1, warning)
        return Group(*panels)

    async def _load_endpoint_records(self) -> tuple[EndpointRecord, ...]:
        if self._endpoint_registry is None:
            return ()
        records = await self._endpoint_registry.list_records()
        return tuple(records)
