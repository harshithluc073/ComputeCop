from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from computecop.cli import app
from computecop.config import EndpointConfig, RuntimeConfig
from computecop.models import EndpointKind, EndpointRoute
from computecop.upstream import HealthProbe


def test_cli_help_imports() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ComputeCop" in result.output


def test_cli_config_prints_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("computecop.cli.load_config", lambda: _config(tmp_path))
    result = CliRunner().invoke(app, ["config"])
    assert result.exit_code == 0
    assert '"server"' in result.output


def test_cli_probe_prints_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("computecop.cli.load_config", lambda: _config(tmp_path))
    monkeypatch.setattr("computecop.cli.build_runtime", lambda config: _fake_runtime())
    result = CliRunner().invoke(app, ["probe"])
    assert result.exit_code == 0
    assert "ollama" in result.output


def test_cli_telemetry_command_runs() -> None:
    result = CliRunner().invoke(app, ["telemetry"])
    assert result.exit_code == 0
    assert "ram_used_percent" in result.output


def _config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        event_log_path=tmp_path / "events.jsonl",
        endpoints=[
            EndpointConfig(
                name="ollama",
                kind=EndpointKind.OLLAMA,
                base_url="http://127.0.0.1:11434",
                health_path="/api/tags",
            )
        ],
    )


def _fake_runtime():
    route = EndpointRoute(
        name="ollama",
        kind=EndpointKind.OLLAMA,
        base_url="http://127.0.0.1:11434",
        timeout_seconds=1.0,
        health_path="/api/tags",
    )

    class FakeUpstream:
        routes = {"ollama": route}

        async def probe(self, route):
            return HealthProbe(
                endpoint=route.name, healthy=False, status_code=None, detail="offline"
            )

        async def close(self):
            return None

    return SimpleNamespace(upstream=FakeUpstream())
