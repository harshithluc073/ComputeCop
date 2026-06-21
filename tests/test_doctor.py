from __future__ import annotations

from pathlib import Path

import pytest

from computecop.config import ConfigError, EffectiveConfig, EndpointConfig, RuntimeConfig
from computecop.doctor import (
    CheckResult,
    CheckStatus,
    DiagnosticReport,
    _check_config_validity,
    _check_endpoint_reachability,
    _check_event_log_path,
    _check_platform_support,
    _check_psutil_access,
    _check_python_version,
    _check_ram_baseline,
    run_diagnostics,
)
from computecop.models import EndpointKind
from computecop.upstream import HealthProbe

GIB = 1024**3


def _effective(tmp_path: Path, endpoints: list[EndpointConfig] | None = None) -> EffectiveConfig:
    config = RuntimeConfig(
        event_log_path=tmp_path / "events.jsonl",
        endpoints=(
            endpoints
            if endpoints is not None
            else [
                EndpointConfig(
                    name="ollama",
                    kind=EndpointKind.OLLAMA,
                    base_url="http://127.0.0.1:11434",
                    health_path="/api/tags",
                )
            ]
        ),
    )
    return EffectiveConfig(config=config, sources={}, config_path=None)


class _FakeRouter:
    def __init__(self, routes: list, *, healthy: bool) -> None:
        self._routes = routes
        self._healthy = healthy
        self.closed = False

    async def probe(self, route):
        return HealthProbe(
            endpoint=route.name,
            healthy=self._healthy,
            status_code=200 if self._healthy else None,
            detail="OK" if self._healthy else "unreachable",
            base_url=route.base_url,
            health_path=route.health_path,
            latency_ms=4.2,
        )

    async def close(self) -> None:
        self.closed = True


def test_python_check_reports_current_interpreter() -> None:
    result = _check_python_version()
    assert result.name == "python"
    assert result.status is CheckStatus.OK
    assert result.detail["required"] == ">=3.11"


def test_platform_check_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("computecop.doctor.current_platform_name", lambda: "linux")
    result = _check_platform_support()
    assert result.status is CheckStatus.OK
    assert result.detail["platform"] == "linux"


def test_platform_check_warns_on_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("computecop.doctor.current_platform_name", lambda: "plan9")
    result = _check_platform_support()
    assert result.status is CheckStatus.WARN
    assert "plan9" in result.summary


def test_ram_check_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("computecop.doctor.total_ram_bytes", lambda: 32 * GIB)
    result = _check_ram_baseline(_effective(tmp_path))
    assert result.status is CheckStatus.OK
    assert result.detail["meets_minimum"] is True


def test_ram_check_warns_below_minimum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("computecop.doctor.total_ram_bytes", lambda: 4 * GIB)
    result = _check_ram_baseline(_effective(tmp_path))
    assert result.status is CheckStatus.WARN
    assert result.detail["meets_minimum"] is False


def test_ram_check_warns_when_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("computecop.doctor.total_ram_bytes", lambda: 0)
    result = _check_ram_baseline(_effective(tmp_path))
    assert result.status is CheckStatus.WARN
    assert result.detail["total_gb"] is None


def test_ram_check_uses_default_minimum_without_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("computecop.doctor.total_ram_bytes", lambda: 8 * GIB)
    result = _check_ram_baseline(None)
    assert result.status is CheckStatus.OK
    assert result.detail["minimum_supported_gb"] == 6.0


def test_psutil_check_ok() -> None:
    result = _check_psutil_access()
    assert result.status is CheckStatus.OK
    assert result.detail["ram_total_bytes"] > 0


def test_psutil_check_fails_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr("computecop.doctor.psutil.virtual_memory", boom)
    result = _check_psutil_access()
    assert result.status is CheckStatus.FAIL
    assert "telemetry unavailable" in result.summary


def test_event_log_check_ok_for_writable_path(tmp_path: Path) -> None:
    result = _check_event_log_path(_effective(tmp_path))
    assert result.status is CheckStatus.OK
    assert result.detail["writable"] is True


def test_event_log_check_warns_when_unwritable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("computecop.doctor.os.access", lambda *_, **__: False)
    result = _check_event_log_path(_effective(tmp_path))
    assert result.status is CheckStatus.WARN
    assert "not writable" in result.summary
    assert "reason" in result.detail


def test_config_check_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    effective = _effective(tmp_path)
    monkeypatch.setattr("computecop.doctor.load_effective_config", lambda **_: effective)
    result, returned = _check_config_validity(None)
    assert result.status is CheckStatus.OK
    assert returned is effective
    assert result.detail["endpoints"] == 1


def test_config_check_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_):
        raise ConfigError("invalid toml")

    monkeypatch.setattr("computecop.doctor.load_effective_config", boom)
    result, returned = _check_config_validity(None)
    assert result.status is CheckStatus.FAIL
    assert returned is None
    assert "invalid toml" in result.detail["error"]


async def test_endpoints_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "computecop.doctor.UpstreamRouter", lambda routes: _FakeRouter(routes, healthy=True)
    )
    result = await _check_endpoint_reachability(_effective(tmp_path), probe_endpoints=True)
    assert result.status is CheckStatus.OK
    assert result.detail["endpoints"][0]["healthy"] is True
    assert result.detail["endpoints"][0]["latency_ms"] == 4.2


async def test_endpoints_warn_when_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "computecop.doctor.UpstreamRouter", lambda routes: _FakeRouter(routes, healthy=False)
    )
    result = await _check_endpoint_reachability(_effective(tmp_path), probe_endpoints=True)
    assert result.status is CheckStatus.WARN
    assert result.detail["probed"] is True


async def test_endpoints_skipped(tmp_path: Path) -> None:
    result = await _check_endpoint_reachability(_effective(tmp_path), probe_endpoints=False)
    assert result.status is CheckStatus.OK
    assert result.detail["probed"] is False


async def test_endpoints_warn_without_config() -> None:
    result = await _check_endpoint_reachability(None, probe_endpoints=True)
    assert result.status is CheckStatus.WARN


async def test_endpoints_fail_when_none_configured(tmp_path: Path) -> None:
    result = await _check_endpoint_reachability(
        _effective(tmp_path, endpoints=[]), probe_endpoints=True
    )
    assert result.status is CheckStatus.FAIL


async def test_run_diagnostics_collects_all_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("computecop.doctor.load_effective_config", lambda **_: _effective(tmp_path))
    report = await run_diagnostics(probe_endpoints=False)
    names = {check.name for check in report.checks}
    assert names == {
        "python",
        "platform",
        "ram",
        "psutil",
        "endpoints",
        "endpoint_capabilities",
        "event_log",
        "port_conflict",
        "config_conflict",
        "thermal",
        "battery",
        "config",
    }
    assert report.ok is True
    assert report.to_dict()["status"] in {"ok", "warn"}


async def test_run_diagnostics_fails_on_invalid_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_):
        raise ConfigError("broken")

    monkeypatch.setattr("computecop.doctor.load_effective_config", boom)
    report = await run_diagnostics(probe_endpoints=False)
    assert report.overall_status is CheckStatus.FAIL
    assert report.ok is False


def test_report_status_precedence() -> None:
    def mk(status: CheckStatus) -> CheckResult:
        return CheckResult(name="x", status=status, summary="", detail={})

    assert DiagnosticReport((mk(CheckStatus.OK),)).overall_status is CheckStatus.OK
    assert (
        DiagnosticReport((mk(CheckStatus.OK), mk(CheckStatus.WARN))).overall_status
        is CheckStatus.WARN
    )
    fail_report = DiagnosticReport((mk(CheckStatus.WARN), mk(CheckStatus.FAIL)))
    assert fail_report.overall_status is CheckStatus.FAIL
    assert fail_report.ok is False
