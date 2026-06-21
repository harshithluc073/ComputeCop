"""Environment diagnostics powering the ``computecop doctor`` command.

The doctor runs a fixed set of read-only checks and aggregates them into a single
report. It is designed so a maintainer can ask a user to run one command and
attach the output: every check returns a machine-readable status plus a short
human summary, and the overall report fails only when a check is genuinely broken
(as opposed to merely degraded, such as an inference engine that is not running).
"""

from __future__ import annotations

import os
import platform as platform_lib
import socket
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import psutil

from computecop.config import ConfigError, EffectiveConfig, load_effective_config
from computecop.events import default_event_log_path
from computecop.models import EndpointRoute
from computecop.platform import HostMemoryProfile, current_platform_name
from computecop.telemetry import total_ram_bytes
from computecop.upstream import UpstreamRouter

MINIMUM_PYTHON: tuple[int, int] = (3, 11)
SUPPORTED_PLATFORMS: tuple[str, ...] = ("windows", "macos", "linux")
DEFAULT_MINIMUM_RAM_GB = 6.0


class CheckStatus(str, Enum):
    """Outcome severity for a single diagnostic check."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class Remediation:
    """Actionable remediation hint for a degraded or failed diagnostic check."""

    severity: str  # "info", "warning", or "error"
    action: str  # Description of what the user should do

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "action": self.action,
        }


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of a single diagnostic check."""

    name: str
    status: CheckStatus
    summary: str
    detail: dict[str, Any]
    remediations: tuple[Remediation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
            "detail": self.detail,
            "remediations": [r.to_dict() for r in self.remediations],
        }


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    """Aggregate of every diagnostic check."""

    checks: tuple[CheckResult, ...]

    @property
    def overall_status(self) -> CheckStatus:
        if any(check.status is CheckStatus.FAIL for check in self.checks):
            return CheckStatus.FAIL
        if any(check.status is CheckStatus.WARN for check in self.checks):
            return CheckStatus.WARN
        return CheckStatus.OK

    @property
    def ok(self) -> bool:
        """Return whether no check hard-failed (warnings are tolerated)."""

        return self.overall_status is not CheckStatus.FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.overall_status.value,
            "checks": [check.to_dict() for check in self.checks],
        }


async def run_diagnostics(
    *,
    config_path: str | Path | None = None,
    probe_endpoints: bool = True,
) -> DiagnosticReport:
    """Run every diagnostic check and return the aggregated report.

    Configuration is loaded first because the endpoint and event-log checks depend
    on it. When configuration is invalid the dependent checks degrade gracefully
    to defaults instead of raising.
    """

    config_check, effective = _check_config_validity(config_path)
    checks = [
        _check_python_version(),
        _check_platform_support(),
        _check_ram_baseline(effective),
        _check_psutil_access(),
        await _check_endpoint_reachability(effective, probe_endpoints=probe_endpoints),
        _check_endpoint_capabilities(effective),
        _check_event_log_path(effective),
        _check_port_conflicts(effective),
        _check_config_source_conflicts(effective),
        _check_thermal_sensors(),
        _check_battery_telemetry(),
        config_check,
    ]
    return DiagnosticReport(checks=tuple(checks))


def _check_python_version() -> CheckResult:
    info = sys.version_info
    current = f"{info.major}.{info.minor}.{info.micro}"
    required = f"{MINIMUM_PYTHON[0]}.{MINIMUM_PYTHON[1]}"
    detail: dict[str, Any] = {"version": current, "required": f">={required}"}
    if (info.major, info.minor) >= MINIMUM_PYTHON:
        return CheckResult(
            name="python",
            status=CheckStatus.OK,
            summary=f"Python {current} satisfies the >= {required} requirement",
            detail=detail,
        )
    return CheckResult(
        name="python",
        status=CheckStatus.FAIL,
        summary=f"Python {current} is older than the required {required}",
        detail=detail,
        remediations=(
            Remediation(
                severity="error",
                action="upgrade Python to 3.11 or newer",
            ),
        ),
    )


def _check_platform_support() -> CheckResult:
    name = current_platform_name()
    detail: dict[str, Any] = {
        "platform": name,
        "system": platform_lib.system(),
        "release": platform_lib.release(),
        "machine": platform_lib.machine(),
        "supported": list(SUPPORTED_PLATFORMS),
    }
    if name in SUPPORTED_PLATFORMS:
        return CheckResult(
            name="platform",
            status=CheckStatus.OK,
            summary=f"{name} is a supported platform",
            detail=detail,
        )
    return CheckResult(
        name="platform",
        status=CheckStatus.WARN,
        summary=f"{name} is not an officially supported platform; behavior is untested",
        detail=detail,
        remediations=(
            Remediation(
                severity="warning",
                action=("run ComputeCop on a supported platform (Windows, macOS, or Linux)"),
            ),
        ),
    )


def _check_ram_baseline(effective: EffectiveConfig | None) -> CheckResult:
    minimum = (
        effective.config.policy.minimum_supported_ram_gb
        if effective is not None
        else DEFAULT_MINIMUM_RAM_GB
    )
    total_bytes = total_ram_bytes()
    if total_bytes <= 0:
        return CheckResult(
            name="ram",
            status=CheckStatus.WARN,
            summary="host RAM capacity could not be determined",
            detail={"total_gb": None, "minimum_supported_gb": minimum},
            remediations=(
                Remediation(
                    severity="warning",
                    action="ensure system has at least the minimum required RAM",
                ),
            ),
        )
    profile = HostMemoryProfile(total_bytes=total_bytes, minimum_supported_gb=minimum)
    detail: dict[str, Any] = {
        "total_gb": round(profile.total_gb, 2),
        "minimum_supported_gb": minimum,
        "meets_minimum": profile.meets_minimum,
    }
    if profile.meets_minimum:
        return CheckResult(
            name="ram",
            status=CheckStatus.OK,
            summary=f"{profile.total_gb:.1f} GiB RAM meets the {minimum:g} GiB minimum",
            detail=detail,
        )
    return CheckResult(
        name="ram",
        status=CheckStatus.WARN,
        summary=f"{profile.total_gb:.1f} GiB RAM is below the {minimum:g} GiB minimum",
        detail=detail,
        remediations=(
            Remediation(
                severity="warning",
                action=(
                    f"upgrade system RAM or close other large applications "
                    f"to meet the {minimum:g} GiB requirement"
                ),
            ),
        ),
    )


def _check_psutil_access() -> CheckResult:
    try:
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
    except (OSError, RuntimeError, AttributeError) as exc:
        return CheckResult(
            name="psutil",
            status=CheckStatus.FAIL,
            summary=f"psutil cannot read host telemetry: {exc}",
            detail={"error": str(exc)},
            remediations=(
                Remediation(
                    severity="error",
                    action=("verify user permissions and ensure psutil has access to system APIs"),
                ),
            ),
        )
    return CheckResult(
        name="psutil",
        status=CheckStatus.OK,
        summary="psutil can read host memory and CPU telemetry",
        detail={
            "ram_total_bytes": int(memory.total),
            "cpu_percent": float(cpu_percent),
        },
    )


async def _check_endpoint_reachability(
    effective: EffectiveConfig | None,
    *,
    probe_endpoints: bool,
) -> CheckResult:
    if effective is None:
        return CheckResult(
            name="endpoints",
            status=CheckStatus.WARN,
            summary="endpoints not checked because configuration is invalid",
            detail={"endpoints": []},
            remediations=(
                Remediation(
                    severity="warning",
                    action=("fix configuration issues to enable endpoint reachability checks"),
                ),
            ),
        )
    routes = [endpoint.to_route() for endpoint in effective.config.endpoints]
    if not routes:
        return CheckResult(
            name="endpoints",
            status=CheckStatus.FAIL,
            summary="no upstream endpoints are configured",
            detail={"endpoints": []},
            remediations=(
                Remediation(
                    severity="error",
                    action="configure at least one upstream endpoint",
                ),
            ),
        )
    if not probe_endpoints:
        return CheckResult(
            name="endpoints",
            status=CheckStatus.OK,
            summary=f"skipped reachability probes for {len(routes)} configured endpoint(s)",
            detail={"probed": False, "endpoints": [_route_summary(route) for route in routes]},
        )
    results = await _probe_routes(routes)
    healthy = [item for item in results if item["healthy"]]
    detail = {"probed": True, "endpoints": results}
    if healthy:
        return CheckResult(
            name="endpoints",
            status=CheckStatus.OK,
            summary=f"{len(healthy)}/{len(results)} configured endpoint(s) reachable",
            detail=detail,
        )
    return CheckResult(
        name="endpoints",
        status=CheckStatus.WARN,
        summary=(
            f"none of {len(results)} configured endpoint(s) are reachable; "
            "start a local inference engine before serving traffic"
        ),
        detail=detail,
        remediations=(
            Remediation(
                severity="warning",
                action=(
                    "start a local inference engine (e.g. Ollama, llama.cpp) before serving traffic"
                ),
            ),
        ),
    )


async def _probe_routes(routes: list[EndpointRoute]) -> list[dict[str, Any]]:
    router = UpstreamRouter(routes)
    results: list[dict[str, Any]] = []
    try:
        for route in routes:
            probe = await router.probe(route)
            results.append(
                {
                    "name": probe.endpoint,
                    "base_url": route.base_url,
                    "healthy": probe.healthy,
                    "status_code": probe.status_code,
                    "latency_ms": (
                        round(probe.latency_ms, 1) if probe.latency_ms is not None else None
                    ),
                    "detail": probe.detail,
                }
            )
    finally:
        await router.close()
    return results


def _route_summary(route: EndpointRoute) -> dict[str, Any]:
    return {"name": route.name, "base_url": route.base_url, "health_path": route.health_path}


def _check_event_log_path(effective: EffectiveConfig | None) -> CheckResult:
    configured = effective.config.event_log_path if effective is not None else None
    path = configured if configured is not None else default_event_log_path()
    writable, reason = _probe_writable(path)
    detail: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "writable": writable,
    }
    if writable:
        return CheckResult(
            name="event_log",
            status=CheckStatus.OK,
            summary=f"event log path is writable: {path}",
            detail=detail,
        )
    detail["reason"] = reason
    return CheckResult(
        name="event_log",
        status=CheckStatus.WARN,
        summary=f"event log path is not writable: {reason}",
        detail=detail,
        remediations=(
            Remediation(
                severity="warning",
                action=(
                    "change the event log path in pyproject.toml / config "
                    "file, or fix file/parent directory permissions"
                ),
            ),
        ),
    )


def _probe_writable(path: Path) -> tuple[bool, str | None]:
    if path.exists():
        if os.access(path, os.W_OK) and os.access(path, os.R_OK):
            return True, None
        return False, f"{path} exists but lacks read/write permissions"
    parent = _first_existing_parent(path)
    if parent is None:
        return False, f"no existing parent directory for {path}"
    if os.access(parent, os.W_OK) and os.access(parent, os.R_OK):
        return True, None
    return False, f"{parent} lacks read/write permissions"


def _first_existing_parent(path: Path) -> Path | None:
    for candidate in path.parents:
        if candidate.exists():
            return candidate
    return None


def _check_config_validity(
    config_path: str | Path | None,
) -> tuple[CheckResult, EffectiveConfig | None]:
    try:
        effective = load_effective_config(config_path=config_path)
    except ConfigError as exc:
        return (
            CheckResult(
                name="config",
                status=CheckStatus.FAIL,
                summary="configuration is invalid",
                detail={
                    "error": str(exc),
                    "config_path": str(config_path) if config_path is not None else None,
                },
                remediations=(
                    Remediation(
                        severity="error",
                        action=(
                            "fix syntax/values in TOML config file or "
                            "verify COMPUTECOP_CONFIG value"
                        ),
                    ),
                ),
            ),
            None,
        )
    source = effective.config_path
    summary = (
        f"configuration loaded from {source}"
        if source is not None
        else "configuration loaded from built-in defaults"
    )
    return (
        CheckResult(
            name="config",
            status=CheckStatus.OK,
            summary=summary,
            detail={
                "config_path": str(source) if source is not None else None,
                "endpoints": len(effective.config.endpoints),
                "log_level": effective.config.log_level,
            },
        ),
        effective,
    )


def _check_thermal_sensors() -> CheckResult:
    """Check if thermal sensors can be read on this system."""
    detail: dict[str, Any] = {
        "supported": hasattr(psutil, "sensors_temperatures"),
    }
    if not hasattr(psutil, "sensors_temperatures"):
        return CheckResult(
            name="thermal",
            status=CheckStatus.WARN,
            summary="thermal sensors are not supported on this platform",
            detail=detail,
            remediations=(
                Remediation(
                    severity="info",
                    action=(
                        "thermal throttling policy will not trigger because sensors are unreadable"
                    ),
                ),
            ),
        )
    try:
        temps = psutil.sensors_temperatures()
        detail["sensors"] = {k: [t._asdict() for t in v] for k, v in temps.items()}
        if not temps:
            return CheckResult(
                name="thermal",
                status=CheckStatus.WARN,
                summary="no thermal sensors detected on host",
                detail=detail,
                remediations=(
                    Remediation(
                        severity="info",
                        action=(
                            "thermal throttling policy will not trigger "
                            "because sensors are unreadable"
                        ),
                    ),
                ),
            )
        all_temps = [t.current for v in temps.values() for t in v]
        max_temp = max(all_temps) if all_temps else None
        detail["max_temp"] = max_temp
        summary_msg = (
            f"thermal sensors are readable (max temp: {max_temp}°C)"
            if max_temp is not None
            else "thermal sensors are readable but returned no values"
        )
        return CheckResult(
            name="thermal",
            status=CheckStatus.OK,
            summary=summary_msg,
            detail=detail,
        )
    except Exception as exc:
        detail["error"] = str(exc)
        return CheckResult(
            name="thermal",
            status=CheckStatus.WARN,
            summary=f"failed to read thermal sensors: {exc}",
            detail=detail,
            remediations=(
                Remediation(
                    severity="info",
                    action=(
                        "thermal throttling policy will not trigger because sensors are unreadable"
                    ),
                ),
            ),
        )


def _check_battery_telemetry() -> CheckResult:
    """Check if battery/power status can be read on this system."""
    detail: dict[str, Any] = {
        "supported": hasattr(psutil, "sensors_battery"),
    }
    if not hasattr(psutil, "sensors_battery"):
        return CheckResult(
            name="battery",
            status=CheckStatus.WARN,
            summary="battery telemetry is not supported on this platform",
            detail=detail,
            remediations=(
                Remediation(
                    severity="info",
                    action=(
                        "battery-aware policy profiles will be inactive "
                        "because battery telemetry is unreadable"
                    ),
                ),
            ),
        )
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            detail["present"] = False
            return CheckResult(
                name="battery",
                status=CheckStatus.OK,
                summary="no battery detected; running on AC power",
                detail=detail,
            )
        detail["present"] = True
        detail["percent"] = battery.percent
        detail["power_plugged"] = battery.power_plugged
        detail["secsleft"] = battery.secsleft
        status_str = "plugged in" if battery.power_plugged else "discharging"
        return CheckResult(
            name="battery",
            status=CheckStatus.OK,
            summary=f"battery detected: {battery.percent}% ({status_str})",
            detail=detail,
        )
    except Exception as exc:
        detail["error"] = str(exc)
        return CheckResult(
            name="battery",
            status=CheckStatus.WARN,
            summary=f"failed to read battery telemetry: {exc}",
            detail=detail,
            remediations=(
                Remediation(
                    severity="info",
                    action=(
                        "battery-aware policy profiles will be inactive "
                        "because battery telemetry is unreadable"
                    ),
                ),
            ),
        )


def _check_port_conflicts(effective: EffectiveConfig | None) -> CheckResult:
    """Check if the configured ComputeCop port is already in use."""
    if effective is None:
        return CheckResult(
            name="port_conflict",
            status=CheckStatus.WARN,
            summary="port check skipped because configuration is invalid",
            detail={},
        )
    host = effective.config.server.host
    port = effective.config.server.port
    detail = {"host": host, "port": port}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        detail["in_use"] = False
        return CheckResult(
            name="port_conflict",
            status=CheckStatus.OK,
            summary=f"port {port} on {host} is free",
            detail=detail,
        )
    except OSError as exc:
        detail["in_use"] = True
        detail["error"] = str(exc)
        return CheckResult(
            name="port_conflict",
            status=CheckStatus.FAIL,
            summary=f"port {port} on {host} is already in use",
            detail=detail,
            remediations=(
                Remediation(
                    severity="error",
                    action=(
                        f"configure a different port in config, or stop "
                        f"the process running on port {port}"
                    ),
                ),
            ),
        )


def _check_config_source_conflicts(effective: EffectiveConfig | None) -> CheckResult:
    """Check if conflicting configuration sources exist (e.g. env variables overriding TOML)."""
    if effective is None:
        return CheckResult(
            name="config_conflict",
            status=CheckStatus.WARN,
            summary="config source check skipped because configuration is invalid",
            detail={},
        )
    from computecop.config import ConfigSource, _leaf_paths, _load_toml_config

    config_path = effective.config_path
    sources = effective.sources
    detail: dict[str, Any] = {
        "config_path": str(config_path) if config_path is not None else None,
        "env_var_config_path": os.environ.get("COMPUTECOP_CONFIG"),
    }
    conflicts: list[dict[str, Any]] = []
    if os.environ.get("COMPUTECOP_CONFIG") and config_path is not None:
        env_path = Path(os.environ["COMPUTECOP_CONFIG"]).expanduser().resolve()
        try:
            resolved_path = config_path.resolve()
        except Exception:
            resolved_path = config_path
        if env_path != resolved_path:
            conflicts.append(
                {
                    "type": "config_file_conflict",
                    "message": (
                        f"CLI config path '{config_path}' overrides "
                        f"COMPUTECOP_CONFIG env var '{env_path}'"
                    ),
                }
            )
    if config_path is not None and config_path.exists():
        try:
            toml_overlay = _load_toml_config(config_path)
            toml_paths = set(_leaf_paths(toml_overlay))
            for path in toml_paths:
                source = sources.get(path)
                if source == ConfigSource.ENVIRONMENT:
                    conflicts.append(
                        {
                            "type": "key_override_conflict",
                            "key": path,
                            "message": (
                                f"key '{path}' defined in TOML is "
                                "overridden by environment variable"
                            ),
                        }
                    )
        except Exception as exc:
            detail["toml_parse_error"] = str(exc)
    detail["conflicts"] = conflicts
    if conflicts:
        summary_msg = f"{len(conflicts)} config source override(s)/conflict(s) detected"
        return CheckResult(
            name="config_conflict",
            status=CheckStatus.WARN,
            summary=summary_msg,
            detail=detail,
            remediations=(
                Remediation(
                    severity="info",
                    action=("clean up conflicting configuration sources to prevent overrides"),
                ),
            ),
        )
    return CheckResult(
        name="config_conflict",
        status=CheckStatus.OK,
        summary="no configuration source conflicts/overrides detected",
        detail=detail,
    )


def _check_endpoint_capabilities(effective: EffectiveConfig | None) -> CheckResult:
    """Report capabilities of all configured endpoints."""
    if effective is None:
        return CheckResult(
            name="endpoint_capabilities",
            status=CheckStatus.WARN,
            summary="endpoint capabilities check skipped because configuration is invalid",
            detail={},
        )
    from computecop.endpoints import EndpointCapabilityRegistry
    from computecop.upstream import UpstreamRouter

    endpoints = effective.config.endpoints
    if not endpoints:
        return CheckResult(
            name="endpoint_capabilities",
            status=CheckStatus.FAIL,
            summary="no endpoints configured to determine capabilities",
            detail={"endpoints": []},
            remediations=(
                Remediation(
                    severity="error",
                    action=("configure at least one upstream endpoint to determine capabilities"),
                ),
            ),
        )
    routes = [endpoint.to_route() for endpoint in endpoints]
    router = UpstreamRouter(routes)
    registry = EndpointCapabilityRegistry(router)
    results: list[dict[str, Any]] = []
    for route in routes:
        caps = registry.capabilities_for(route)
        results.append(
            {
                "name": route.name,
                "kind": route.kind.value,
                "supports_streaming": caps.supports_streaming,
                "supports_model_list": caps.supports_model_list,
                "supports_offload": caps.supports_offload,
                "default_context_tokens": caps.default_context_tokens,
                "default_output_tokens": caps.default_output_tokens,
            }
        )
    return CheckResult(
        name="endpoint_capabilities",
        status=CheckStatus.OK,
        summary=f"retrieved capabilities for {len(results)} configured endpoint(s)",
        detail={"endpoints": results},
    )
