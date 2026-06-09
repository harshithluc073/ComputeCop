"""Command line entrypoints for ComputeCop."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from computecop.app import build_runtime, create_app
from computecop.config import ConfigError, EffectiveConfig, load_config, load_effective_config
from computecop.dashboard import Dashboard
from computecop.doctor import CheckStatus, DiagnosticReport, run_diagnostics
from computecop.events import (
    JsonlEventStore,
    event_matches_correlation,
    summarize_events,
)
from computecop.logging import configure_logging
from computecop.models import to_jsonable
from computecop.shutdown import ShutdownCoordinator, cancel_task
from computecop.telemetry import PsutilTelemetrySampler
from computecop.upstream import HealthProbe

app = typer.Typer(
    name="computecop",
    help="Local inference traffic controller with telemetry-aware compute budgeting.",
    no_args_is_help=True,
)
config_app = typer.Typer(help="Inspect effective ComputeCop configuration.")
app.add_typer(config_app, name="config")
events_app = typer.Typer(help="Inspect persisted ComputeCop runtime events.")
app.add_typer(events_app, name="events")


class CliContext:
    """Shared CLI state for config path resolution."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help="Path to a TOML configuration file.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    """Run ComputeCop commands."""

    ctx.ensure_object(dict)
    ctx.obj["cli"] = CliContext(config_path=config)


@config_app.callback(invoke_without_command=True)
def print_config(ctx: typer.Context) -> None:
    """Print the effective runtime configuration."""

    if ctx.invoked_subcommand is not None:
        return
    config = _load_or_exit(ctx)
    Console().print_json(json.dumps(to_jsonable(config)))


@config_app.command("explain")
def explain_config(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit the config explanation as JSON.",
    ),
) -> None:
    """Explain effective configuration values and their sources."""

    effective = _load_effective_or_exit(ctx)
    if json_output:
        Console().print_json(json.dumps(effective.explain_document()))
        return

    table = Table(title="ComputeCop Configuration Sources")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_column("Source")
    for entry in effective.explain_entries():
        table.add_row(entry["path"], _format_explain_value(entry["value"]), entry["source"])
    Console().print(table)
    if effective.config_path is not None:
        Console().print(f"[dim]Config file: {effective.config_path}[/dim]")


@events_app.command("tail")
def events_tail(
    ctx: typer.Context,
    limit: int = typer.Option(
        20, "--limit", "-n", min=1, max=1000, help="Number of recent events to show."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit events as JSON."),
) -> None:
    """Show the most recent persisted runtime events."""

    store = _event_store(ctx)
    events = list(asyncio.run(store.tail(limit=limit)))
    if json_output:
        Console().print_json(json.dumps({"events": events}))
        return
    _print_events_table(events, title=f"Recent Events ({len(events)})")


@events_app.command("find")
def events_find(
    ctx: typer.Context,
    correlation_id: Annotated[
        str,
        typer.Option("--correlation-id", help="Correlation or trace ID to match."),
    ],
    limit: int = typer.Option(
        100, "--limit", "-n", min=1, max=1000, help="Maximum matching events to show."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit matching events as JSON."),
) -> None:
    """Find persisted events by correlation or trace ID."""

    store = _event_store(ctx)
    events = asyncio.run(store.read_events())
    matched = [event for event in events if event_matches_correlation(event, correlation_id)]
    matched = matched[-limit:]
    if json_output:
        Console().print_json(json.dumps({"correlation_id": correlation_id, "events": matched}))
        return
    if not matched:
        Console().print(f"[yellow]no events found for correlation id '{correlation_id}'[/yellow]")
        return
    _print_events_table(matched, title=f"Events for {correlation_id} ({len(matched)})")


@events_app.command("stats")
def events_stats(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit statistics as JSON."),
) -> None:
    """Summarize persisted runtime events by kind and time range."""

    store = _event_store(ctx)
    events = asyncio.run(store.read_events())
    stats = summarize_events(events)
    if json_output:
        Console().print_json(json.dumps(stats))
        return
    table = Table(title="ComputeCop Event Statistics")
    table.add_column("Kind")
    table.add_column("Count", justify="right")
    by_kind: dict[str, int] = stats["by_kind"]
    for kind, count in by_kind.items():
        table.add_row(kind, str(count))
    if not by_kind:
        table.add_row("none", "0")
    console = Console()
    console.print(table)
    console.print(
        f"total={stats['total']} "
        f"earliest={stats['earliest'] or '-'} "
        f"latest={stats['latest'] or '-'}"
    )


@app.command()
def run(
    ctx: typer.Context,
    host: str | None = typer.Option(None, help="Host to bind."),
    port: int | None = typer.Option(None, min=1, max=65535, help="Port to bind."),
    log_level: str | None = typer.Option(None, help="Logging level."),
) -> None:
    """Run the ComputeCop proxy server."""

    cli_overrides = _cli_overrides(host=host, port=port, log_level=log_level)
    config = _load_or_exit(ctx, cli_overrides=cli_overrides)
    configure_logging(config.log_level)
    bind_host = host or config.server.host
    bind_port = port or config.server.port
    if bind_host not in {"127.0.0.1", "localhost", "::1"} and not config.server.expose_remote:
        raise typer.BadParameter("remote exposure requires COMPUTECOP_EXPOSE_REMOTE=true")
    uvicorn.run(
        create_app(config), host=bind_host, port=bind_port, log_level=config.log_level.lower()
    )


@app.command()
def dashboard(ctx: typer.Context) -> None:
    """Run the live terminal dashboard."""

    config = _load_or_exit(ctx)
    configure_logging(config.log_level, rich=True)
    runtime = build_runtime(config)
    shutdown = ShutdownCoordinator()

    async def _run_dashboard() -> None:
        await runtime.start()
        dashboard_task = asyncio.create_task(Dashboard(runtime.state).run())
        try:
            await dashboard_task
        except asyncio.CancelledError:
            shutdown.request_shutdown()
            raise
        finally:
            await cancel_task(dashboard_task)
            await shutdown.shutdown_runtime(runtime)

    try:
        asyncio.run(_run_dashboard())
    except KeyboardInterrupt:
        if shutdown.request_shutdown():
            Console().print("[yellow]ComputeCop dashboard stopped[/yellow]")
    except asyncio.CancelledError:
        if shutdown.request_shutdown():
            Console().print("[yellow]ComputeCop dashboard stopped[/yellow]")


@app.command()
def telemetry() -> None:
    """Print a one-shot telemetry sample."""

    async def _sample() -> None:
        sample = await PsutilTelemetrySampler().sample()
        Console().print_json(json.dumps(to_jsonable(sample)))

    asyncio.run(_sample())


@app.command()
def probe(ctx: typer.Context) -> None:
    """Probe configured upstream inference endpoints."""

    config = _load_or_exit(ctx)
    runtime = build_runtime(config)

    async def _probe() -> None:
        table = Table(title="ComputeCop Endpoint Probes")
        table.add_column("Endpoint")
        table.add_column("Healthy")
        table.add_column("Status")
        table.add_column("Latency")
        table.add_column("Failures")
        table.add_column("Last OK")
        table.add_column("Detail")
        try:
            for route in runtime.upstream.routes.values():
                result = await runtime.upstream.probe(route)
                table.add_row(
                    result.endpoint,
                    "yes" if result.healthy else "no",
                    _probe_status(result),
                    _probe_latency(result.latency_ms),
                    str(result.failure_streak),
                    _probe_last_success(result.last_success_at),
                    result.detail,
                )
        finally:
            await runtime.upstream.close()
        Console().print(table)

    try:
        asyncio.run(_probe())
    except KeyboardInterrupt:
        Console().print("[yellow]ComputeCop probe stopped[/yellow]")


@app.command()
def doctor(
    ctx: typer.Context,
    json_output: bool = typer.Option(False, "--json", help="Emit diagnostics as JSON."),
    skip_endpoints: bool = typer.Option(
        False,
        "--skip-endpoints",
        help="Skip upstream endpoint reachability probes.",
    ),
) -> None:
    """Run environment diagnostics and report ComputeCop readiness."""

    config_path = _cli_context(ctx).config_path
    report = asyncio.run(
        run_diagnostics(config_path=config_path, probe_endpoints=not skip_endpoints)
    )
    if json_output:
        Console().print_json(json.dumps(report.to_dict()))
    else:
        _print_doctor_report(report)
    raise typer.Exit(code=0 if report.ok else 1)


def _print_doctor_report(report: DiagnosticReport) -> None:
    table = Table(title="ComputeCop Doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Summary")
    for check in report.checks:
        table.add_row(check.name, _status_label(check.status), check.summary)
    console = Console()
    console.print(table)
    console.print(f"overall: {_status_label(report.overall_status)}")


def _status_label(status: CheckStatus) -> str:
    color = {
        CheckStatus.OK: "green",
        CheckStatus.WARN: "yellow",
        CheckStatus.FAIL: "red",
    }[status]
    return f"[{color}]{status.value}[/{color}]"


def _probe_status(result: HealthProbe) -> str:
    if result.status_code is not None:
        return str(result.status_code)
    if result.failure_category is not None:
        return result.failure_category.value
    return "-"


def _probe_latency(latency_ms: float | None) -> str:
    return f"{latency_ms:.0f}ms" if latency_ms is not None else "-"


def _probe_last_success(last_success_at: datetime | None) -> str:
    return last_success_at.strftime("%H:%M:%S") if last_success_at is not None else "never"


def _cli_context(ctx: typer.Context) -> CliContext:
    ctx.ensure_object(dict)
    cli = ctx.obj.get("cli")
    return cli if isinstance(cli, CliContext) else CliContext()


def _event_store(ctx: typer.Context) -> JsonlEventStore:
    return JsonlEventStore(_load_or_exit(ctx).event_log_path)


def _print_events_table(events: list[dict[str, Any]], *, title: str) -> None:
    table = Table(title=title)
    table.add_column("Time")
    table.add_column("Kind")
    table.add_column("Detail")
    for event in events:
        table.add_row(_event_time(event), str(event.get("kind", "-")), _event_detail(event))
    if not events:
        table.add_row("-", "none", "no events recorded")
    Console().print(table)


def _event_time(event: dict[str, Any]) -> str:
    timestamp = event.get("timestamp")
    return str(timestamp) if timestamp else "-"


def _event_detail(event: dict[str, Any]) -> str:
    payload = event.get("payload")
    if isinstance(payload, dict) and payload:
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return "-"


def _load_effective_or_exit(ctx: typer.Context) -> EffectiveConfig:
    try:
        return load_effective_config(config_path=_cli_context(ctx).config_path)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_or_exit(ctx: typer.Context, *, cli_overrides: dict[str, Any] | None = None):
    try:
        return load_config(
            config_path=_cli_context(ctx).config_path,
            cli_overrides=cli_overrides,
        )
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _cli_overrides(
    *,
    host: str | None,
    port: int | None,
    log_level: str | None,
) -> dict[str, Any] | None:
    overrides: dict[str, Any] = {}
    server: dict[str, Any] = {}
    if host is not None:
        server["host"] = host
    if port is not None:
        server["port"] = port
    if server:
        overrides["server"] = server
    if log_level is not None:
        overrides["log_level"] = log_level.upper()
    return overrides or None


def _format_explain_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)
