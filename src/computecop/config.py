"""Configuration loading and validation for ComputeCop."""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from computecop.models import EndpointKind, EndpointRoute, to_jsonable

CONFIG_ENV_VAR = "COMPUTECOP_CONFIG"


class ConfigError(RuntimeError):
    """Raised when ComputeCop configuration is invalid."""


class ConfigSource(str, Enum):
    """Origin of an effective configuration value."""

    DEFAULT = "default"
    TOML = "toml"
    ENVIRONMENT = "environment"
    CLI = "cli"


class EndpointConfig(BaseModel):
    """User-configurable upstream endpoint."""

    name: str
    kind: EndpointKind
    base_url: str
    timeout_seconds: float = Field(default=120.0, ge=1.0, le=3600.0)
    health_path: str = "/"
    supports_streaming: bool = True

    @field_validator("base_url")
    @classmethod
    def normalize_base_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return cleaned

    @field_validator("health_path")
    @classmethod
    def normalize_health_path(cls, value: str) -> str:
        cleaned = value.strip() or "/"
        return cleaned if cleaned.startswith("/") else f"/{cleaned}"

    def to_route(self) -> EndpointRoute:
        return EndpointRoute(
            name=self.name,
            kind=self.kind,
            base_url=self.base_url,
            timeout_seconds=self.timeout_seconds,
            health_path=self.health_path,
            supports_streaming=self.supports_streaming,
        )


class TelemetryConfig(BaseModel):
    """Telemetry sampler settings."""

    interval_seconds: float = Field(default=1.0, ge=0.2, le=30.0)
    smoothing_window: int = Field(default=5, ge=1, le=120)
    heavy_process_limit: int = Field(default=12, ge=0, le=100)
    disk_sample_interval_seconds: float = Field(default=1.0, ge=0.2, le=30.0)


class PolicyConfig(BaseModel):
    """Juice and pressure thresholds."""

    ram_yield_percent: float = Field(default=85.0, ge=50.0, le=99.0)
    ram_recover_percent: float = Field(default=78.0, ge=30.0, le=98.0)
    ram_recover_gap_percent: float = Field(default=7.0, ge=2.0, le=30.0)
    minimum_supported_ram_gb: float = Field(default=6.0, ge=2.0, le=64.0)
    cpu_pressure_percent: float = Field(default=88.0, ge=10.0, le=100.0)
    swap_pressure_percent: float = Field(default=30.0, ge=0.0, le=100.0)
    thermal_warm_celsius: float = Field(default=75.0, ge=30.0, le=110.0)
    thermal_hot_celsius: float = Field(default=88.0, ge=40.0, le=115.0)
    thermal_critical_celsius: float = Field(default=95.0, ge=50.0, le=125.0)
    foreground_juice_level: int = Field(default=100, ge=1, le=100)
    background_base_juice_level: int = Field(default=70, ge=1, le=100)
    minimum_background_juice_level: int = Field(default=10, ge=1, le=100)
    base_context_tokens: int = Field(default=8192, ge=512, le=262144)
    base_output_tokens: int = Field(default=2048, ge=32, le=32768)
    max_background_concurrency: int = Field(default=2, ge=1, le=32)
    max_foreground_concurrency: int = Field(default=4, ge=1, le=64)

    @field_validator("ram_recover_percent")
    @classmethod
    def recover_below_yield(cls, value: float, info: Any) -> float:
        yield_percent = info.data.get("ram_yield_percent")
        if yield_percent is not None and value >= yield_percent:
            raise ValueError("ram_recover_percent must be lower than ram_yield_percent")
        return value


class EndpointRegistryConfig(BaseModel):
    """Endpoint capability registry settings."""

    capability_probe_ttl_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=600.0,
        description="Seconds to cache endpoint capability health probes.",
    )
    health_watcher_enabled: bool = Field(
        default=True,
        description="Run a background loop that proactively probes endpoint health.",
    )
    health_watcher_interval_seconds: float = Field(
        default=15.0,
        ge=1.0,
        le=600.0,
        description="Base interval between background endpoint health probe cycles.",
    )
    health_watcher_jitter_fraction: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Random jitter fraction applied to the health watcher interval.",
    )
    circuit_breaker_failure_threshold: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Consecutive failures before an endpoint circuit breaker opens.",
    )
    circuit_breaker_cooldown_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=3600.0,
        description="Seconds to wait before a circuit breaker enters half-open.",
    )
    circuit_breaker_half_open_successes: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Successful probes or requests required to close a half-open breaker.",
    )


class QueueConfig(BaseModel):
    """Async request queue limits."""

    max_size: int = Field(default=128, ge=1, le=10000)
    default_timeout_seconds: float = Field(default=900.0, ge=1.0, le=86400.0)
    background_retry_after_seconds: float = Field(default=3.0, ge=0.1, le=3600.0)
    shutdown_drain_seconds: float = Field(default=5.0, ge=0.0, le=300.0)
    aging_interval_seconds: float = Field(
        default=30.0,
        ge=0.1,
        le=600.0,
        description="Seconds of queue wait before a background item gains scheduling priority.",
    )


class ServerConfig(BaseModel):
    """Proxy server binding and exposure settings."""

    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    expose_remote: bool = False


class RuntimeConfig(BaseModel):
    """Complete ComputeCop runtime configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    endpoint_registry: EndpointRegistryConfig = Field(default_factory=EndpointRegistryConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    queue: QueueConfig = Field(default_factory=QueueConfig)
    endpoints: list[EndpointConfig] = Field(
        default_factory=lambda: [
            EndpointConfig(
                name="ollama",
                kind=EndpointKind.OLLAMA,
                base_url="http://127.0.0.1:11434",
                health_path="/api/tags",
            ),
            EndpointConfig(
                name="llama-cpp",
                kind=EndpointKind.LLAMA_CPP,
                base_url="http://127.0.0.1:8080",
                health_path="/health",
            ),
        ]
    )
    event_log_path: Path | None = None
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def uppercase_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
            raise ValueError("log_level must be one of CRITICAL, ERROR, WARNING, INFO, DEBUG")
        return normalized

    def route_for_name(self, name: str | None = None) -> EndpointRoute:
        if not self.endpoints:
            raise ConfigError("at least one endpoint must be configured")
        if name is None:
            return self.endpoints[0].to_route()
        for endpoint in self.endpoints:
            if endpoint.name == name:
                return endpoint.to_route()
        known = ", ".join(endpoint.name for endpoint in self.endpoints)
        raise ConfigError(f"unknown endpoint '{name}', configured endpoints: {known}")


@dataclass(slots=True)
class EffectiveConfig:
    """Runtime configuration with per-field source metadata."""

    config: RuntimeConfig
    sources: dict[str, ConfigSource]
    config_path: Path | None = None

    def explain_entries(self) -> list[dict[str, Any]]:
        """Return flattened config values with their effective sources."""

        payload = to_jsonable(self.config)
        entries: list[dict[str, Any]] = []
        for path in sorted(_leaf_paths(payload)):
            entries.append(
                {
                    "path": path,
                    "value": _value_at_path(payload, path),
                    "source": self.sources.get(path, ConfigSource.DEFAULT).value,
                }
            )
        return entries

    def explain_document(self) -> dict[str, Any]:
        """Return a JSON-safe config explanation document."""

        return {
            "config_path": str(self.config_path) if self.config_path is not None else None,
            "entries": self.explain_entries(),
        }


def resolve_config_path(
    config_path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve an optional TOML config file path."""

    env = environ or os.environ
    if config_path is not None:
        path = Path(config_path).expanduser()
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        return path

    raw = env.get(CONFIG_ENV_VAR)
    if raw is None or raw.strip() == "":
        return None

    path = Path(raw).expanduser()
    if not path.is_file():
        raise ConfigError(f"{CONFIG_ENV_VAR} points to missing file: {path}")
    return path


def load_effective_config(
    *,
    config_path: str | Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> EffectiveConfig:
    """Load configuration with deterministic precedence and source metadata."""

    env = environ or os.environ
    resolved_path = resolve_config_path(config_path, environ=env)
    data = _default_config_data()
    sources = {path: ConfigSource.DEFAULT for path in _leaf_paths(data)}

    if resolved_path is not None:
        toml_overlay = _load_toml_config(resolved_path)
        data = _deep_merge(data, toml_overlay)
        for path in _leaf_paths(toml_overlay):
            sources[path] = ConfigSource.TOML

    env_overlay = _env_config_overlay(env)
    if env_overlay:
        data = _deep_merge(data, env_overlay)
        for path in _leaf_paths(env_overlay):
            sources[path] = ConfigSource.ENVIRONMENT

    if cli_overrides:
        cli_overlay = dict(cli_overrides)
        data = _deep_merge(data, cli_overlay)
        for path in _leaf_paths(cli_overlay):
            sources[path] = ConfigSource.CLI

    try:
        config = RuntimeConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc

    return EffectiveConfig(config=config, sources=sources, config_path=resolved_path)


def load_config(
    *,
    config_path: str | Path | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> RuntimeConfig:
    """Load ComputeCop configuration from defaults, TOML, env, and CLI overrides."""

    return load_effective_config(
        config_path=config_path,
        cli_overrides=cli_overrides,
        environ=environ,
    ).config


@lru_cache(maxsize=1)
def cached_config() -> RuntimeConfig:
    """Return a cached runtime config for dependency injection."""

    return load_config()


def _default_config_data() -> dict[str, Any]:
    return RuntimeConfig().model_dump(mode="json")


def _load_toml_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file must contain a TOML table: {path}")
    return data


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _leaf_paths(data: Any, prefix: str = "") -> set[str]:
    if isinstance(data, dict):
        paths: set[str] = set()
        for key, value in data.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict) and value:
                paths.update(_leaf_paths(value, child))
            else:
                paths.add(child)
        return paths
    return {prefix} if prefix else set()


def _value_at_path(data: Any, path: str) -> Any:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _json_env(name: str, environ: Mapping[str, str]) -> dict[str, Any] | list[Any] | None:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{name} must contain valid JSON") from exc
    if not isinstance(parsed, (dict, list)):
        raise ConfigError(f"{name} must contain a JSON object or list")
    return parsed


def _optional_str_env(name: str, environ: Mapping[str, str]) -> str | None:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def _optional_float_env(name: str, environ: Mapping[str, str]) -> float | None:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float") from exc


def _optional_int_env(name: str, environ: Mapping[str, str]) -> int | None:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _optional_bool_env(name: str, environ: Mapping[str, str]) -> bool | None:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_config_overlay(environ: Mapping[str, str]) -> dict[str, Any]:
    overlay: dict[str, Any] = {}

    server: dict[str, Any] = {}
    if (host := _optional_str_env("COMPUTECOP_HOST", environ)) is not None:
        server["host"] = host
    if (port := _optional_int_env("COMPUTECOP_PORT", environ)) is not None:
        server["port"] = port
    if (expose_remote := _optional_bool_env("COMPUTECOP_EXPOSE_REMOTE", environ)) is not None:
        server["expose_remote"] = expose_remote
    if server:
        overlay["server"] = server

    telemetry: dict[str, Any] = {}
    if (interval := _optional_float_env("COMPUTECOP_TELEMETRY_INTERVAL", environ)) is not None:
        telemetry["interval_seconds"] = interval
    if (smoothing := _optional_int_env("COMPUTECOP_SMOOTHING_WINDOW", environ)) is not None:
        telemetry["smoothing_window"] = smoothing
    if (heavy_limit := _optional_int_env("COMPUTECOP_HEAVY_PROCESS_LIMIT", environ)) is not None:
        telemetry["heavy_process_limit"] = heavy_limit
    if telemetry:
        overlay["telemetry"] = telemetry

    policy: dict[str, Any] = {}
    policy_fields = {
        "ram_yield_percent": "COMPUTECOP_RAM_YIELD_PERCENT",
        "ram_recover_percent": "COMPUTECOP_RAM_RECOVER_PERCENT",
        "ram_recover_gap_percent": "COMPUTECOP_RAM_RECOVER_GAP_PERCENT",
        "minimum_supported_ram_gb": "COMPUTECOP_MIN_RAM_GB",
        "cpu_pressure_percent": "COMPUTECOP_CPU_PRESSURE_PERCENT",
        "swap_pressure_percent": "COMPUTECOP_SWAP_PRESSURE_PERCENT",
    }
    for field_name, env_name in policy_fields.items():
        if (value := _optional_float_env(env_name, environ)) is not None:
            policy[field_name] = value
    if policy:
        overlay["policy"] = policy

    queue: dict[str, Any] = {}
    if (max_size := _optional_int_env("COMPUTECOP_QUEUE_MAX_SIZE", environ)) is not None:
        queue["max_size"] = max_size
    if (timeout := _optional_float_env("COMPUTECOP_QUEUE_TIMEOUT", environ)) is not None:
        queue["default_timeout_seconds"] = timeout
    if queue:
        overlay["queue"] = queue

    if (log_level := _optional_str_env("COMPUTECOP_LOG_LEVEL", environ)) is not None:
        overlay["log_level"] = log_level

    if (event_log := _optional_str_env("COMPUTECOP_EVENT_LOG", environ)) is not None:
        overlay["event_log_path"] = str(Path(event_log).expanduser())

    endpoint_data = _json_env("COMPUTECOP_ENDPOINTS", environ)
    if endpoint_data is not None:
        if not isinstance(endpoint_data, list):
            raise ConfigError("COMPUTECOP_ENDPOINTS must be a JSON list")
        overlay["endpoints"] = endpoint_data

    return overlay
