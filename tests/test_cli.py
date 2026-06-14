from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from computecop.cli import app
from computecop.config import (
    ConfigError,
    ConfigSource,
    EffectiveConfig,
    EndpointConfig,
    RuntimeConfig,
)
from computecop.models import EndpointKind, EndpointRoute
from computecop.upstream import HealthProbe, UpstreamFailureCategory


def test_cli_help_imports() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ComputeCop" in result.output


def test_cli_config_prints_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("computecop.cli.load_config", lambda **_: _config(tmp_path))
    result = CliRunner().invoke(app, ["config"])
    assert result.exit_code == 0
    assert '"server"' in result.output


def test_cli_config_explain_prints_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "computecop.cli.load_effective_config", lambda **_: _effective_config(tmp_path)
    )
    result = CliRunner().invoke(app, ["config", "explain"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "Configuration Sources" in result.output
    assert "server.port" in result.output
    assert "default" in result.output


def test_cli_config_explain_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "computecop.cli.load_effective_config", lambda **_: _effective_config(tmp_path)
    )
    result = CliRunner().invoke(app, ["config", "explain", "--json"])
    assert result.exit_code == 0
    assert '"entries"' in result.output
    assert '"server.port"' in result.output


def test_cli_probe_prints_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("computecop.cli.load_config", lambda **_: _config(tmp_path))
    monkeypatch.setattr("computecop.cli.build_runtime", lambda config: _fake_runtime())
    result = CliRunner().invoke(app, ["probe"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "ollama" in result.output
    assert "Latency" in result.output
    assert "unreachable" in result.output
    assert "never" in result.output


def test_cli_telemetry_command_runs() -> None:
    result = CliRunner().invoke(app, ["telemetry"])
    assert result.exit_code == 0
    assert "ram_used_percent" in result.output


def test_cli_events_tail_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_events(tmp_path / "events.jsonl")
    monkeypatch.setattr("computecop.cli.load_config", lambda **_: _config(tmp_path))
    result = CliRunner().invoke(app, ["events", "tail", "-n", "2"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "Recent Events" in result.output
    assert "policy.yield" in result.output
    assert "upstream.failure" in result.output
    # The oldest event is trimmed by the limit.
    assert "admission.decision" not in result.output


def test_cli_events_tail_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_events(tmp_path / "events.jsonl")
    monkeypatch.setattr("computecop.cli.load_config", lambda **_: _config(tmp_path))
    result = CliRunner().invoke(app, ["events", "tail", "--json"])
    assert result.exit_code == 0
    assert '"events"' in result.output
    assert '"policy.yield"' in result.output


def test_cli_events_find_by_correlation_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_events(tmp_path / "events.jsonl")
    monkeypatch.setattr("computecop.cli.load_config", lambda **_: _config(tmp_path))
    result = CliRunner().invoke(app, ["events", "find", "--correlation-id", "corr-123", "--json"])
    assert result.exit_code == 0
    # Matches the top-level correlation_id and the one nested in the decision payload.
    assert '"admission.decision"' in result.output
    assert '"upstream.failure"' in result.output
    assert '"policy.yield"' not in result.output


def test_cli_events_find_by_trace_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_events(tmp_path / "events.jsonl")
    monkeypatch.setattr("computecop.cli.load_config", lambda **_: _config(tmp_path))
    result = CliRunner().invoke(app, ["events", "find", "--correlation-id", "t-abc", "--json"])
    assert result.exit_code == 0
    assert '"admission.decision"' in result.output
    assert '"upstream.failure"' not in result.output


def test_cli_events_find_not_found(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_events(tmp_path / "events.jsonl")
    monkeypatch.setattr("computecop.cli.load_config", lambda **_: _config(tmp_path))
    result = CliRunner().invoke(app, ["events", "find", "--correlation-id", "missing"])
    assert result.exit_code == 0
    assert "no events found" in result.output


def test_cli_events_stats_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_events(tmp_path / "events.jsonl")
    monkeypatch.setattr("computecop.cli.load_config", lambda **_: _config(tmp_path))
    result = CliRunner().invoke(app, ["events", "stats"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "Event Statistics" in result.output
    assert "admission.decision" in result.output
    assert "total=3" in result.output


def test_cli_events_stats_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_events(tmp_path / "events.jsonl")
    monkeypatch.setattr("computecop.cli.load_config", lambda **_: _config(tmp_path))
    result = CliRunner().invoke(app, ["events", "stats", "--json"])
    assert result.exit_code == 0
    assert '"total": 3' in result.output
    assert '"by_kind"' in result.output


def test_cli_events_stats_empty_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("computecop.cli.load_config", lambda **_: _config(tmp_path))
    result = CliRunner().invoke(app, ["events", "stats"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "total=0" in result.output


def test_cli_doctor_table(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "computecop.doctor.load_effective_config", lambda **_: _effective_config(tmp_path)
    )
    result = CliRunner().invoke(app, ["doctor", "--skip-endpoints"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "ComputeCop Doctor" in result.output
    assert "overall:" in result.output


def test_cli_doctor_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "computecop.doctor.load_effective_config", lambda **_: _effective_config(tmp_path)
    )
    result = CliRunner().invoke(app, ["doctor", "--skip-endpoints", "--json"])
    assert result.exit_code == 0
    assert '"checks"' in result.output
    assert '"python"' in result.output
    assert '"config"' in result.output


def test_cli_doctor_fails_on_invalid_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_):
        raise ConfigError("broken config")

    monkeypatch.setattr("computecop.doctor.load_effective_config", boom)
    result = CliRunner().invoke(app, ["doctor", "--skip-endpoints", "--json"])
    assert result.exit_code == 1
    assert '"fail"' in result.output


def _write_events(path: Path) -> None:
    rows = [
        {
            "kind": "admission.decision",
            "timestamp": "2026-06-09T10:00:00+00:00",
            "payload": {
                "path": "/v1/chat/completions",
                "trace_id": "t-abc",
                "decision": {"correlation_id": "corr-123"},
            },
        },
        {
            "kind": "policy.yield",
            "timestamp": "2026-06-09T10:01:00+00:00",
            "payload": {"reason": "ram pressure"},
        },
        {
            "kind": "upstream.failure",
            "timestamp": "2026-06-09T10:02:00+00:00",
            "payload": {"correlation_id": "corr-123", "category": "timeout"},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


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


def _effective_config(tmp_path: Path) -> EffectiveConfig:
    config = _config(tmp_path)
    return EffectiveConfig(
        config=config,
        sources={"server.port": ConfigSource.DEFAULT},
        config_path=tmp_path / "computecop.toml",
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
                endpoint=route.name,
                healthy=False,
                status_code=None,
                detail="endpoint 'ollama' is unreachable",
                base_url=route.base_url,
                health_path=route.health_path,
                latency_ms=12.5,
                failure_category=UpstreamFailureCategory.UNREACHABLE,
                failure_streak=3,
                last_success_at=None,
            )

        async def close(self):
            return None

    return SimpleNamespace(upstream=FakeUpstream())


def test_cli_queue_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("computecop.cli.load_config", lambda **_: _config(tmp_path))

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"ok": True, "state": "paused"}

    class FakeResponseResume:
        status_code = 200

        def json(self):
            return {"ok": True, "state": "accepting"}

    def fake_post(url, **kwargs):
        if "pause" in url:
            return FakeResponse()
        return FakeResponseResume()

    monkeypatch.setattr("httpx.post", fake_post)

    result_pause = CliRunner().invoke(app, ["queue", "pause"])
    assert result_pause.exit_code == 0
    assert "Successfully paused" in result_pause.output

    result_resume = CliRunner().invoke(app, ["queue", "resume"])
    assert result_resume.exit_code == 0
    assert "Successfully resumed" in result_resume.output


def test_cli_profiles_list() -> None:
    # 1. Standard list
    result = CliRunner().invoke(app, ["profiles", "list"])
    assert result.exit_code == 0
    assert "Built-In Profiles" in result.output
    assert "low-memory" in result.output
    assert "battery-saver" in result.output

    # 2. JSON list
    result_json = CliRunner().invoke(app, ["profiles", "list", "--json"])
    assert result_json.exit_code == 0
    parsed = json.loads(result_json.output)
    assert "low-memory" in parsed["profiles"]
    assert "balanced" in parsed["profiles"]


def test_cli_profiles_show() -> None:
    # 1. Balanced profile show (no overrides)
    result_bal = CliRunner().invoke(app, ["profiles", "show", "balanced"])
    assert result_bal.exit_code == 0
    assert "balanced" in result_bal.output
    assert "system defaults" in result_bal.output

    # 2. Specific profile show (shows table of values)
    result_lm = CliRunner().invoke(app, ["profiles", "show", "low-memory"])
    assert result_lm.exit_code == 0
    assert "low-memory" in result_lm.output
    assert "base_context_tokens" in result_lm.output
    assert "4096" in result_lm.output

    # 3. JSON output show
    result_json = CliRunner().invoke(app, ["profiles", "show", "low-memory", "--json"])
    assert result_json.exit_code == 0
    parsed = json.loads(result_json.output)
    assert parsed["policy"]["base_context_tokens"] == 4096

    # 4. Invalid profile name show
    result_err = CliRunner().invoke(app, ["profiles", "show", "invalid-profile"])
    assert result_err.exit_code == 1
    assert "Unknown profile name" in result_err.output


def test_cli_run_with_profile_option(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured_config = None

    def fake_create_app(config):
        nonlocal captured_config
        captured_config = config
        return "fake_app"

    def fake_uvicorn_run(app, **kwargs):
        pass

    monkeypatch.setattr("computecop.cli.create_app", fake_create_app)
    monkeypatch.setattr("uvicorn.run", fake_uvicorn_run)
    monkeypatch.setattr("computecop.cli.configure_logging", lambda *_, **__: None)

    result = CliRunner().invoke(app, ["run", "--profile", "low-memory"])
    assert result.exit_code == 0
    assert captured_config is not None
    assert captured_config.profile.value == "low-memory"
    assert captured_config.policy.max_background_concurrency == 1
    assert captured_config.policy.base_context_tokens == 4096
