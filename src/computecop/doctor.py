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
class CheckResult:
    """Result of a single diagnostic check."""

    name: str
    status: CheckStatus
    summary: str
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
            "detail": self.detail,
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
        _check_event_log_path(effective),
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
        )
    routes = [endpoint.to_route() for endpoint in effective.config.endpoints]
    if not routes:
        return CheckResult(
            name="endpoints",
            status=CheckStatus.FAIL,
            summary="no upstream endpoints are configured",
            detail={"endpoints": []},
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
    )


def _probe_writable(path: Path) -> tuple[bool, str | None]:
    if path.exists():
        if os.access(path, os.W_OK):
            return True, None
        return False, f"{path} exists but is not writable"
    parent = _first_existing_parent(path)
    if parent is None:
        return False, f"no existing parent directory for {path}"
    if os.access(parent, os.W_OK):
        return True, None
    return False, f"{parent} is not writable"


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
