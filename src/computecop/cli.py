"""Command line entrypoints for ComputeCop."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

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


@app.callback()
def main() -> None:
    """Run ComputeCop commands."""


@app.command()
def run(
    host: str | None = typer.Option(None, help="Host to bind."),
    port: int | None = typer.Option(None, min=1, max=65535, help="Port to bind."),
    log_level: str | None = typer.Option(None, help="Logging level."),
) -> None:
    """Run the ComputeCop proxy server."""

    config = _load_or_exit()
    if log_level:
        config.log_level = log_level.upper()
    configure_logging(config.log_level)
    bind_host = host or config.server.host
    bind_port = port or config.server.port
    if bind_host not in {"127.0.0.1", "localhost", "::1"} and not config.server.expose_remote:
        raise typer.BadParameter("remote exposure requires COMPUTECOP_EXPOSE_REMOTE=true")
    uvicorn.run(
        create_app(config), host=bind_host, port=bind_port, log_level=config.log_level.lower()
    )


@app.command()
def dashboard() -> None:
    """Run the live terminal dashboard."""

    config = _load_or_exit()
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


@app.command("config")
def print_config() -> None:
    """Print the effective runtime configuration."""

    config = _load_or_exit()
    Console().print_json(json.dumps(to_jsonable(config)))


@app.command()
def telemetry() -> None:
    """Print a one-shot telemetry sample."""

    async def _sample() -> None:
        sample = await PsutilTelemetrySampler().sample()
        Console().print_json(json.dumps(to_jsonable(sample)))

    asyncio.run(_sample())


@app.command()
def probe() -> None:
    """Probe configured upstream inference endpoints."""

    config = _load_or_exit()
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


def _load_or_exit():
    try:
        return load_config()
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
