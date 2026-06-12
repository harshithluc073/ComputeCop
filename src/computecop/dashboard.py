"""Rich terminal dashboard for ComputeCop."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from computecop.models import TelemetrySample
from computecop.state import RuntimeSnapshot, RuntimeStateStore
from computecop.telemetry import format_bytes_per_second


class Dashboard:
    """Render a minimal live terminal dashboard."""

    def __init__(self, state: RuntimeStateStore, refresh_seconds: float = 0.5) -> None:
        self.state = state
        self.refresh_seconds = refresh_seconds

    async def run(self) -> None:
        """Run until cancelled."""

        with Live(await self.render(), refresh_per_second=4, screen=False) as live:
            while True:
                await asyncio.sleep(self.refresh_seconds)
                live.update(await self.render())

    async def render(self) -> Group:
        snapshot = await self.state.snapshot()
        panels = [
            self._header(snapshot),
            self._resource_panel(snapshot.telemetry),
            self._policy_panel(snapshot),
            self._worker_panel(snapshot),
            self._trace_panel(snapshot),
            self._decision_panel(snapshot),
        ]
        warning = self._persistence_warning(snapshot)
        if warning is not None:
            panels.insert(1, warning)
        return Group(*panels)

    def _persistence_warning(self, snapshot: RuntimeSnapshot) -> Panel | None:
        persistence = snapshot.event_persistence
        if persistence.enabled:
            return None
        reason = persistence.disabled_reason or "unknown error"
        line = Text.assemble(
            ("Event persistence disabled", "bold red"),
            "  ",
            (reason, "red"),
        )
        return Panel(line, title="Warning", border_style="red")

    def _header(self, snapshot: RuntimeSnapshot) -> Panel:
        title = Text("ComputeCop", style="bold cyan")
        status = "YIELD" if snapshot.yield_active else snapshot.system_state.value.upper()
        color = "red" if snapshot.yield_active else "green"
        line = Text.assemble(title, "  ", (status, f"bold {color}"), "  ")
        line.append(f"juice={snapshot.global_juice_level}")
        return Panel(Align.center(line), border_style=color)

    def _resource_panel(self, telemetry: TelemetrySample | None) -> Panel:
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        if telemetry is None:
            table.add_row(
                "CPU: collecting", "RAM: collecting", "Thermal: unknown", "Disk: collecting"
            )
            return Panel(table, title="Resources", border_style="blue")

        table.add_row(
            f"CPU: {telemetry.cpu_percent:.1f}%",
            f"RAM: {telemetry.ram_used_percent:.1f}% ({telemetry.ram_available_gb:.1f} GiB free)",
            f"Thermal: {telemetry.thermal_state.value}",
            (
                f"Disk: R {format_bytes_per_second(telemetry.disk_read_bytes_per_sec)} / "
                f"W {format_bytes_per_second(telemetry.disk_write_bytes_per_sec)}"
            ),
        )
        process_table = Table(title="Heavy Processes", expand=True)
        process_table.add_column("PID", justify="right")
        process_table.add_column("Name")
        process_table.add_column("CPU", justify="right")
        process_table.add_column("RSS", justify="right")
        for process in telemetry.heavy_processes[:6]:
            process_table.add_row(
                str(process.pid),
                process.name[:28],
                f"{process.cpu_percent:.1f}%",
                f"{process.memory_rss_mb:.0f} MiB",
            )
        if not telemetry.heavy_processes:
            process_table.add_row("-", "none detected", "-", "-")
        return Panel(Group(table, process_table), title="Resources", border_style="blue")

    def _policy_panel(self, snapshot: RuntimeSnapshot) -> Panel:
        table = Table(expand=True)
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("System state", snapshot.system_state.value)
        table.add_row("Yield active", str(snapshot.yield_active))
        table.add_row("Yield reason", snapshot.yield_reason or "-")
        table.add_row("Queue state", snapshot.queue.lifecycle_state.value)
        table.add_row("Queued", str(snapshot.queue.queued))
        table.add_row("Running foreground", str(snapshot.queue.running_foreground))
        table.add_row("Running background", str(snapshot.queue.running_background))
        table.add_row("Completed", str(snapshot.queue.completed))
        table.add_row("Rejected", str(snapshot.queue.rejected))
        scheduler = snapshot.scheduler
        table.add_row("Scheduler foreground slots", str(scheduler.reserved_foreground_slots))
        table.add_row("Scheduler background slots", str(scheduler.effective_background_slots))
        table.add_row(
            "Scheduler running fg/bg",
            f"{scheduler.running_foreground}/{scheduler.running_background}",
        )
        table.add_row("Scheduler spare slots", str(scheduler.spare_slots))
        concurrency = snapshot.concurrency
        if concurrency is not None:
            table.add_row(
                "Endpoint fg/bg limits",
                (
                    f"{concurrency.limits.max_endpoint_foreground}/"
                    f"{concurrency.limits.max_endpoint_background}"
                ),
            )
            if concurrency.endpoints:
                endpoint = concurrency.endpoints[0]
                table.add_row(
                    f"Endpoint {endpoint.endpoint_name} running",
                    f"{endpoint.running_foreground}/{endpoint.running_background}",
                )
        return Panel(table, title="Policy", border_style="magenta")

    def _worker_panel(self, snapshot: RuntimeSnapshot) -> Panel:
        table = Table(expand=True)
        table.add_column("Worker")
        table.add_column("State")
        table.add_column("Correlation")
        for worker in snapshot.queue.workers:
            correlation = worker.active_correlation_id or "-"
            table.add_row(worker.worker_id, worker.state.value, correlation[:24])
        if not snapshot.queue.workers:
            table.add_row("-", "none registered", "-")
        return Panel(table, title="Queue Workers", border_style="green")

    def _trace_panel(self, snapshot: RuntimeSnapshot) -> Panel:
        latest = snapshot.recent_decisions[0] if snapshot.recent_decisions else None
        trace = latest.trace if latest is not None else None
        table = Table(expand=True)
        table.add_column("Rule")
        table.add_column("Observed")
        table.add_column("Threshold")
        table.add_column("Penalty", justify="right")
        table.add_column("Detail")

        if trace is None:
            table.add_row("-", "-", "-", "-", "no policy trace observed")
            return Panel(table, title="Why", border_style="yellow")

        for rule in trace.rules[:8]:
            table.add_row(
                rule.name,
                "-" if rule.observed is None else str(rule.observed),
                "-" if rule.threshold is None else str(rule.threshold),
                str(rule.penalty),
                rule.detail[:72],
            )
        if not trace.rules:
            table.add_row("-", "-", "-", "-", trace.summary)
        return Panel(table, title=f"Why: {trace.summary[:64]}", border_style="yellow")

    def _decision_panel(self, snapshot: RuntimeSnapshot) -> Panel:
        table = Table(expand=True)
        table.add_column("When")
        table.add_column("Decision")
        table.add_column("Class")
        table.add_column("Juice", justify="right")
        table.add_column("Reason")
        for decision in snapshot.recent_decisions[:8]:
            table.add_row(
                _relative_time(decision.budget.reason),
                decision.decision.value,
                decision.request_class.value,
                str(decision.budget.juice_level),
                decision.reason[:60],
            )
        if not snapshot.recent_decisions:
            table.add_row("-", "none", "-", "-", "no requests observed")
        return Panel(table, title="Recent Decisions", border_style="cyan")


def _relative_time(_: str) -> str:
    return datetime.now(tz=UTC).strftime("%H:%M:%S")
