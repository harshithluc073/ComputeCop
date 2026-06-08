"""Command line entrypoints for ComputeCop."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from computecop.app import build_runtime, create_app
from computecop.config import ConfigError, load_config
from computecop.dashboard import Dashboard
from computecop.logging import configure_logging
from computecop.models import to_jsonable
from computecop.telemetry import PsutilTelemetrySampler
from computecop.upstream import HealthProbe

app = typer.Typer(
    name="computecop",
    help="Local inference traffic controller with telemetry-aware compute budgeting.",
    no_args_is_help=True,
)
class CliContext:
    """Shared CLI state for config path resolution."""

    def __init__(self, config_path: Path | None = None) -> None:
        self.config_path = config_path


@app.callback()
def main(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Path to a TOML configuration file.",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
) -> None:
    """Run ComputeCop commands."""

    ctx.ensure_object(dict)
    ctx.obj["cli"] = CliContext(config_path=config)


@app.command("config")
def print_config(ctx: typer.Context) -> None:
    """Print the effective runtime configuration."""

    config = _load_or_exit(ctx)
    Console().print_json(json.dumps(to_jsonable(config)))


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

    async def _run_dashboard() -> None:
        await runtime.start()
        try:
            await Dashboard(runtime.state).run()
        finally:
            await runtime.stop()

    try:
        asyncio.run(_run_dashboard())
    except KeyboardInterrupt:
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

    asyncio.run(_probe())


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