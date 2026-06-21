"""Modular Rich panel renderers for the ComputeCop dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from rich.align import Align
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from computecop.endpoints import EndpointRecord
from computecop.models import TelemetrySample
from computecop.state import RuntimeSnapshot
from computecop.telemetry import format_bytes_per_second


@dataclass(frozen=True, slots=True)
class DashboardLayout:
    """Stable panel dimensions for dashboard v2."""

    panel_width: int = 120
    header_height: int = 3
    resource_height: int = 12
    policy_height: int = 13
    scheduler_height: int = 12
    endpoint_height: int = 10
    worker_height: int = 8
    trace_height: int = 10
    decision_height: int = 12
    footer_height: int = 6


DEFAULT_LAYOUT = DashboardLayout()


def panel(
    renderable: RenderableType,
    *,
    title: str,
    border_style: str,
    layout: DashboardLayout = DEFAULT_LAYOUT,
    height: int | None = None,
) -> Panel:
    """Wrap a renderable in a consistently sized dashboard panel."""

    return Panel(
        renderable,
        title=title,
        border_style=border_style,
        width=layout.panel_width,
        height=height,
    )


def render_header(snapshot: RuntimeSnapshot, *, layout: DashboardLayout = DEFAULT_LAYOUT) -> Panel:
    title = Text("ComputeCop", style="bold cyan")
    status = "YIELD" if snapshot.yield_active else snapshot.system_state.value.upper()
    color = "red" if snapshot.yield_active else "green"
    line = Text.assemble(title, "  ", (status, f"bold {color}"), "  ")
    line.append(f"juice={snapshot.global_juice_level}")
    return panel(
        Align.center(line),
        title="Status",
        border_style=color,
        layout=layout,
        height=layout.header_height,
    )


def render_persistence_warning(
    snapshot: RuntimeSnapshot,
    *,
    layout: DashboardLayout = DEFAULT_LAYOUT,
) -> Panel | None:
    persistence = snapshot.event_persistence
    if persistence.enabled:
        return None
    reason = persistence.disabled_reason or "unknown error"
    line = Text.assemble(
        ("Event persistence disabled", "bold red"),
        "  ",
        (reason, "red"),
    )
    return panel(line, title="Warning", border_style="red", layout=layout)


def render_footer(
    *,
    detail_mode: bool = False,
    pending_action: str | None = None,
    status_message: str | None = None,
    draining: bool = False,
    layout: DashboardLayout = DEFAULT_LAYOUT,
) -> Panel:
    shortcuts = "[P]ause  [R]esume  [D]rain  [T]oggle detail  [Q]uit"
    if pending_action == "drain":
        shortcuts += "  |  Confirm: [D]  Cancel: [C]"
    header = Text(shortcuts, style="dim")
    detail = Text(f"Detail: {'ON' if detail_mode else 'OFF'}", style="cyan")
    status = Text()
    if draining:
        status.append("Draining queue...", "bold yellow")
    if status_message:
        if draining:
            status.append("  ")
        status.append(status_message, "yellow")
    rows = [Align.center(header), Align.center(detail)]
    if status.plain:
        rows.append(Align.center(status))
    body = Group(*rows)
    return panel(
        body,
        title="Controls",
        border_style="dim",
        layout=layout,
        height=layout.footer_height,
    )


def render_resource_panel(
    snapshot: RuntimeSnapshot,
    *,
    layout: DashboardLayout = DEFAULT_LAYOUT,
    detail_mode: bool = False,
) -> Panel:
    telemetry = snapshot.telemetry
    table = Table.grid(expand=True)
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    table.add_column(ratio=1)
    if telemetry is None:
        table.add_row(
            "CPU: collecting",
            "RAM: collecting",
            "Thermal: unknown",
            "Disk: collecting",
        )
        process_table = Table(title="Heavy Processes", expand=True)
        process_table.add_column("PID", justify="right")
        process_table.add_column("Name")
        process_table.add_column("CPU", justify="right")
        process_table.add_column("RSS", justify="right")
        process_table.add_row("-", "awaiting telemetry", "-", "-")
        return panel(
            Group(table, process_table),
            title="Resources",
            border_style="blue",
            layout=layout,
            height=layout.resource_height,
        )

    table.add_row(
        f"CPU: {telemetry.cpu_percent:.1f}%",
        f"RAM: {telemetry.ram_used_percent:.1f}% ({telemetry.ram_available_gb:.1f} GiB free)",
        f"Thermal: {telemetry.thermal_state.value}",
        (
            f"Disk: R {format_bytes_per_second(telemetry.disk_read_bytes_per_sec)} / "
            f"W {format_bytes_per_second(telemetry.disk_write_bytes_per_sec)}"
        ),
    )
    process_table = _heavy_process_table(telemetry, detail_mode=detail_mode)
    return panel(
        Group(table, process_table),
        title="Resources",
        border_style="blue",
        layout=layout,
        height=layout.resource_height,
    )


def _heavy_process_table(telemetry: TelemetrySample, *, detail_mode: bool = False) -> Table:
    process_table = Table(title="Heavy Processes", expand=True)
    process_table.add_column("PID", justify="right")
    process_table.add_column("Name")
    process_table.add_column("CPU", justify="right")
    process_table.add_column("RSS", justify="right")
    process_limit = 10 if detail_mode else 6
    for process in telemetry.heavy_processes[:process_limit]:
        process_table.add_row(
            str(process.pid),
            process.name[:28],
            f"{process.cpu_percent:.1f}%",
            f"{process.memory_rss_mb:.0f} MiB",
        )
    if not telemetry.heavy_processes:
        process_table.add_row("-", "none detected", "-", "-")
    return process_table


def render_policy_panel(
    snapshot: RuntimeSnapshot,
    *,
    layout: DashboardLayout = DEFAULT_LAYOUT,
) -> Panel:
    table = Table(expand=True)
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("System state", snapshot.system_state.value)
    table.add_row("Yield active", str(snapshot.yield_active))
    table.add_row("Yield reason", snapshot.yield_reason or "-")
    table.add_row("Global juice", str(snapshot.global_juice_level))
    concurrency = snapshot.concurrency
    if concurrency is not None:
        table.add_row(
            "Endpoint fg/bg limits",
            (
                f"{concurrency.limits.max_endpoint_foreground}/"
                f"{concurrency.limits.max_endpoint_background}"
            ),
        )
        table.add_row(
            "Global fg/bg limits",
            f"{concurrency.limits.max_foreground}/{concurrency.limits.max_background}",
        )
        if concurrency.limits.reasons:
            table.add_row("Limit reasons", ", ".join(concurrency.limits.reasons[:3]))
    else:
        table.add_row("Concurrency limits", "not initialized")
    return panel(
        table,
        title="Policy",
        border_style="magenta",
        layout=layout,
        height=layout.policy_height,
    )


def render_scheduler_panel(
    snapshot: RuntimeSnapshot,
    *,
    layout: DashboardLayout = DEFAULT_LAYOUT,
) -> Panel:
    table = Table(expand=True)
    table.add_column("Metric")
    table.add_column("Value")
    queue = snapshot.queue
    scheduler = snapshot.scheduler
    table.add_row("Queue state", queue.lifecycle_state.value)
    table.add_row("Queued", str(queue.queued))
    table.add_row("Running foreground", str(queue.running_foreground))
    table.add_row("Running background", str(queue.running_background))
    table.add_row("Completed", str(queue.completed))
    table.add_row("Rejected", str(queue.rejected))
    table.add_row("Scheduler foreground slots", str(scheduler.reserved_foreground_slots))
    table.add_row("Scheduler background slots", str(scheduler.effective_background_slots))
    table.add_row(
        "Scheduler running fg/bg",
        f"{scheduler.running_foreground}/{scheduler.running_background}",
    )
    table.add_row("Scheduler spare slots", str(scheduler.spare_slots))
    table.add_row("Immediate executions", str(scheduler.immediate_executions))
    table.add_row("Queued executions", str(scheduler.queued_executions))
    return panel(
        table,
        title="Scheduler",
        border_style="bright_magenta",
        layout=layout,
        height=layout.scheduler_height,
    )


def render_endpoint_panel(
    snapshot: RuntimeSnapshot,
    endpoints: tuple[EndpointRecord, ...],
    *,
    layout: DashboardLayout = DEFAULT_LAYOUT,
) -> Panel:
    table = Table(expand=True)
    table.add_column("Endpoint")
    table.add_column("Health")
    table.add_column("Breaker")
    table.add_column("Latency", justify="right")
    table.add_column("Running fg/bg", justify="right")
    capacity_by_name = _endpoint_capacity(snapshot)
    if not endpoints:
        table.add_row("-", "none configured", "-", "-", "-")
        return panel(
            table,
            title="Endpoints",
            border_style="cyan",
            layout=layout,
            height=layout.endpoint_height,
        )

    for record in endpoints[:8]:
        health_label = "healthy" if record.health.healthy else "unhealthy"
        breaker = (
            record.health.circuit_breaker.state.value
            if record.health.circuit_breaker is not None
            else "closed"
        )
        latency = (
            f"{record.health.latency_ms:.0f} ms" if record.health.latency_ms is not None else "-"
        )
        running = capacity_by_name.get(record.name, (0, 0))
        table.add_row(
            record.name,
            health_label,
            breaker,
            latency,
            f"{running[0]}/{running[1]}",
        )
    return panel(
        table,
        title="Endpoints",
        border_style="cyan",
        layout=layout,
        height=layout.endpoint_height,
    )


def _endpoint_capacity(snapshot: RuntimeSnapshot) -> dict[str, tuple[int, int]]:
    concurrency = snapshot.concurrency
    if concurrency is None:
        return {}
    return {
        endpoint.endpoint_name: (
            endpoint.running_foreground,
            endpoint.running_background,
        )
        for endpoint in concurrency.endpoints
    }


def render_worker_panel(
    snapshot: RuntimeSnapshot,
    *,
    layout: DashboardLayout = DEFAULT_LAYOUT,
) -> Panel:
    table = Table(expand=True)
    table.add_column("Worker")
    table.add_column("State")
    table.add_column("Correlation")
    for worker in snapshot.queue.workers:
        correlation = worker.active_correlation_id or "-"
        table.add_row(worker.worker_id, worker.state.value, correlation[:24])
    if not snapshot.queue.workers:
        table.add_row("-", "none registered", "-")
    return panel(
        table,
        title="Queue Workers",
        border_style="green",
        layout=layout,
        height=layout.worker_height,
    )


def render_trace_panel(
    snapshot: RuntimeSnapshot,
    *,
    layout: DashboardLayout = DEFAULT_LAYOUT,
    detail_mode: bool = False,
) -> Panel:
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
        return panel(
            table,
            title="Policy Trace",
            border_style="yellow",
            layout=layout,
            height=layout.trace_height,
        )

    rule_limit = 16 if detail_mode else 8
    detail_width = 120 if detail_mode else 72
    for rule in trace.rules[:rule_limit]:
        table.add_row(
            rule.name,
            "-" if rule.observed is None else str(rule.observed),
            "-" if rule.threshold is None else str(rule.threshold),
            str(rule.penalty),
            rule.detail[:detail_width],
        )
    if not trace.rules:
        table.add_row("-", "-", "-", "-", trace.summary)
    return panel(
        table,
        title=f"Policy Trace: {trace.summary[:64]}",
        border_style="yellow",
        layout=layout,
        height=layout.trace_height,
    )


def render_decision_panel(
    snapshot: RuntimeSnapshot,
    *,
    layout: DashboardLayout = DEFAULT_LAYOUT,
    detail_mode: bool = False,
) -> Panel:
    table = Table(expand=True)
    table.add_column("When")
    table.add_column("Decision")
    table.add_column("Class")
    table.add_column("Juice", justify="right")
    table.add_column("Reason")
    decision_limit = 16 if detail_mode else 8
    reason_width = 100 if detail_mode else 60
    for decision in snapshot.recent_decisions[:decision_limit]:
        table.add_row(
            _relative_time(decision.budget.reason),
            decision.decision.value,
            decision.request_class.value,
            str(decision.budget.juice_level),
            decision.reason[:reason_width],
        )
    if not snapshot.recent_decisions:
        table.add_row("-", "none", "-", "-", "no requests observed")
    return panel(
        table,
        title="Recent Decisions",
        border_style="bright_cyan",
        layout=layout,
        height=layout.decision_height,
    )


def _relative_time(_: str) -> str:
    return datetime.now(tz=UTC).strftime("%H:%M:%S")
