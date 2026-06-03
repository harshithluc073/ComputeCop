"""Command line entrypoints for ComputeCop."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="computecop",
    help="Local inference traffic controller with telemetry-aware compute budgeting.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Run ComputeCop commands."""

