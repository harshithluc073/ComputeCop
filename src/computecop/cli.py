"""Command line entrypoints for ComputeCop."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from computecop.app import build_runtime, create_app
from computecop.config import (
    PROFILES,
    ConfigError,
    EffectiveConfig,
    ProfileName,
    load_config,
    load_effective_config,
)
from computecop.dashboard import Dashboard
from computecop.dashboard_controls import DashboardQueueController
from computecop.doctor import CheckStatus, DiagnosticReport, Remediation, run_diagnostics
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
queue_app = typer.Typer(help="Control and inspect the background request queue.")
app.add_typer(queue_app, name="queue")
profiles_app = typer.Typer(help="Manage and inspect policy profiles.")
app.add_typer(profiles_app, name="profiles")


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
    profile: str | None = typer.Option(None, help="Policy profile to run with."),
) -> None:
    """Run the ComputeCop proxy server."""

    cli_overrides = _cli_overrides(host=host, port=port, log_level=log_level, profile=profile)
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
        queue_controller = DashboardQueueController(
            runtime.queue,
            drain_seconds=config.queue.shutdown_drain_seconds,
        )
        dashboard_task = asyncio.create_task(
            Dashboard(
                runtime.state,
                endpoint_registry=runtime.endpoint_registry,
                queue_controller=queue_controller,
            ).run()
        )
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

    remediations: list[tuple[str, Remediation]] = []
    for check in report.checks:
        for r in check.remediations:
            remediations.append((check.name, r))
    if remediations:
        console.print("\n[bold]Remediation Hints:[/bold]")
        for name, r in remediations:
            color = {
                "info": "blue",
                "warning": "yellow",
                "error": "red",
            }.get(r.severity, "white")
            console.print(f"  • [[{color}]{r.severity}[/{color}]] {name}: {r.action}")


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
    profile: str | None = None,
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
    if profile is not None:
        overrides["profile"] = profile
    return overrides or None


def _format_explain_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


@queue_app.command("pause")
def queue_pause(ctx: typer.Context) -> None:
    """Pause the background request queue."""

    _send_queue_command(ctx, "pause")


@queue_app.command("resume")
def queue_resume(ctx: typer.Context) -> None:
    """Resume the background request queue."""

    _send_queue_command(ctx, "resume")


def _send_queue_command(ctx: typer.Context, command: str) -> None:
    config = _load_or_exit(ctx)
    host = config.server.host
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    url = f"http://{host}:{config.server.port}/queue/{command}"

    console = Console()
    try:
        response = httpx.post(url, timeout=5.0)
        if response.status_code == 200:
            res_data = response.json()
            if res_data.get("ok"):
                action = "paused" if command == "pause" else "resumed"
                state = res_data.get("state")
                console.print(
                    f"[green]Successfully {action} the queue. Current state: {state}[/green]"
                )
            else:
                console.print(f"[red]Failed to {command} the queue.[/red]")
                raise typer.Exit(code=1) from None
        else:
            console.print(
                f"[red]Error from server (HTTP {response.status_code}): {response.text}[/red]"
            )
            raise typer.Exit(code=1) from None
    except Exception as exc:
        console.print(f"[red]Could not connect to ComputeCop daemon at {url}: {exc}[/red]")
        raise typer.Exit(code=1) from None


@profiles_app.command("list")
def profiles_list(
    json_output: bool = typer.Option(False, "--json", help="Emit profiles as JSON."),
) -> None:
    """List all available policy profiles."""

    if json_output:
        profile_names = [p.value for p in ProfileName]
        Console().print_json(json.dumps({"profiles": profile_names}))
        return

    table = Table(title="ComputeCop Built-In Profiles")
    table.add_column("Profile")
    table.add_column("Description")

    descriptions = {
        ProfileName.BALANCED: "Balanced defaults for general workloads.",
        ProfileName.FOREGROUND_FIRST: "Prioritizes user prompts, limits background concurrency.",
        ProfileName.BACKGROUND_THROUGHPUT: (
            "Optimizes background agent throughput and limits concurrency less aggressively."
        ),
        ProfileName.BATTERY_SAVER: (
            "Conserves power by limiting CPU pressure threshold and concurrency."
        ),
        ProfileName.THERMAL_SAFE: "Prevents overheating by lowering thermal pressure thresholds.",
        ProfileName.LOW_MEMORY: (
            "Optimizes for low RAM hosts by shrinking token budgets and limits."
        ),
    }

    for name in ProfileName:
        table.add_row(name.value, descriptions.get(name, "-"))

    Console().print(table)


@profiles_app.command("show")
def profiles_show(
    name: str = typer.Argument(..., help="Name of the profile to inspect."),
    json_output: bool = typer.Option(False, "--json", help="Emit profile configuration as JSON."),
) -> None:
    """Show the configuration details for a specific profile."""

    try:
        profile_name = ProfileName(name)
    except ValueError as exc:
        Console().print(f"[red]Error: Unknown profile name '{name}'[/red]")
        raise typer.Exit(code=1) from exc

    overlay = PROFILES[profile_name]

    if json_output:
        Console().print_json(json.dumps(overlay))
        return

    if not overlay:
        Console().print(f"[bold]{profile_name.value}[/bold] (Balanced)")
        Console().print("Uses all system defaults.")
        return

    table = Table(title=f"Profile Details: {profile_name.value}")
    table.add_column("Category")
    table.add_column("Setting")
    table.add_column("Value")

    for category, settings in overlay.items():
        for key, val in settings.items():
            table.add_row(category, key, str(val))

    Console().print(table)
