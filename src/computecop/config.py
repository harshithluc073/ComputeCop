"""Configuration loading and validation for ComputeCop."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from computecop.models import EndpointKind, EndpointRoute


class ConfigError(RuntimeError):
    """Raised when ComputeCop configuration is invalid."""


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


class QueueConfig(BaseModel):
    """Async request queue limits."""

    max_size: int = Field(default=128, ge=1, le=10000)
    default_timeout_seconds: float = Field(default=900.0, ge=1.0, le=86400.0)
    background_retry_after_seconds: float = Field(default=3.0, ge=0.1, le=3600.0)


class ServerConfig(BaseModel):
    """Proxy server binding and exposure settings."""

    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    expose_remote: bool = False


class RuntimeConfig(BaseModel):
    """Complete ComputeCop runtime configuration."""

    server: ServerConfig = Field(default_factory=ServerConfig)
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


def _json_env(name: str) -> dict[str, Any] | list[Any] | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{name} must contain valid JSON") from exc
    if not isinstance(parsed, (dict, list)):
        raise ConfigError(f"{name} must contain a JSON object or list")
    return parsed


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a float") from exc


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _str_env(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


def load_config() -> RuntimeConfig:
    """Load ComputeCop configuration from defaults and environment variables."""

    endpoint_data = _json_env("COMPUTECOP_ENDPOINTS")
    if endpoint_data is not None and not isinstance(endpoint_data, list):
        raise ConfigError("COMPUTECOP_ENDPOINTS must be a JSON list")

    event_log = os.getenv("COMPUTECOP_EVENT_LOG")
    data: dict[str, Any] = {
        "server": {
            "host": _str_env("COMPUTECOP_HOST", "127.0.0.1"),
            "port": _int_env("COMPUTECOP_PORT", 8765),
            "expose_remote": _str_env("COMPUTECOP_EXPOSE_REMOTE", "false").lower()
            in {"1", "true", "yes", "on"},
        },
        "telemetry": {
            "interval_seconds": _float_env("COMPUTECOP_TELEMETRY_INTERVAL", 1.0),
            "smoothing_window": _int_env("COMPUTECOP_SMOOTHING_WINDOW", 5),
            "heavy_process_limit": _int_env("COMPUTECOP_HEAVY_PROCESS_LIMIT", 12),
        },
        "policy": {
            "ram_yield_percent": _float_env("COMPUTECOP_RAM_YIELD_PERCENT", 85.0),
            "ram_recover_percent": _float_env("COMPUTECOP_RAM_RECOVER_PERCENT", 78.0),
            "ram_recover_gap_percent": _float_env("COMPUTECOP_RAM_RECOVER_GAP_PERCENT", 7.0),
            "minimum_supported_ram_gb": _float_env("COMPUTECOP_MIN_RAM_GB", 6.0),
            "cpu_pressure_percent": _float_env("COMPUTECOP_CPU_PRESSURE_PERCENT", 88.0),
            "swap_pressure_percent": _float_env("COMPUTECOP_SWAP_PRESSURE_PERCENT", 30.0),
        },
        "queue": {
            "max_size": _int_env("COMPUTECOP_QUEUE_MAX_SIZE", 128),
            "default_timeout_seconds": _float_env("COMPUTECOP_QUEUE_TIMEOUT", 900.0),
        },
        "log_level": _str_env("COMPUTECOP_LOG_LEVEL", "INFO"),
        "event_log_path": Path(event_log).expanduser() if event_log else None,
    }
    if endpoint_data is not None:
        data["endpoints"] = endpoint_data

    try:
        return RuntimeConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


@lru_cache(maxsize=1)
def cached_config() -> RuntimeConfig:
    """Return a cached runtime config for dependency injection."""

    return load_config()
