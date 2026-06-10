"""Endpoint capability registry with TTL-cached health probing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from computecop.health import CircuitBreakerRegistry, CircuitBreakerStatus
from computecop.models import EndpointKind, EndpointRoute, utc_now
from computecop.upstream import HealthProbe, UpstreamRouter

_FAMILY_ALIASES: dict[str, EndpointKind] = {
    "ollama": EndpointKind.OLLAMA,
    "llama_cpp": EndpointKind.LLAMA_CPP,
    "llama-cpp": EndpointKind.LLAMA_CPP,
    "openai": EndpointKind.OPENAI_COMPATIBLE,
    "openai_compatible": EndpointKind.OPENAI_COMPATIBLE,
}

_DEFAULT_CONTEXT_TOKENS: dict[EndpointKind, int] = {
    EndpointKind.OLLAMA: 8192,
    EndpointKind.LLAMA_CPP: 8192,
    EndpointKind.OPENAI_COMPATIBLE: 8192,
}

_DEFAULT_OUTPUT_TOKENS: dict[EndpointKind, int] = {
    EndpointKind.OLLAMA: 2048,
    EndpointKind.LLAMA_CPP: 2048,
    EndpointKind.OPENAI_COMPATIBLE: 2048,
}


@dataclass(frozen=True, slots=True)
class EndpointCapabilities:
    """Static and inferred capabilities for a configured upstream endpoint."""

    api_family: EndpointKind
    supports_streaming: bool
    supports_model_list: bool
    supports_offload: bool
    default_context_tokens: int
    default_output_tokens: int


@dataclass(frozen=True, slots=True)
class EndpointHealthStatus:
    """Cached health snapshot for an endpoint."""

    healthy: bool
    status_code: int | None
    latency_ms: float | None
    failure_rate: float
    failure_streak: int
    last_success_at: datetime | None
    checked_at: datetime | None
    detail: str
    status_category: str | None = None
    stale: bool = False
    circuit_breaker: CircuitBreakerStatus | None = None


@dataclass(frozen=True, slots=True)
class EndpointRoutingMetadata:
    """Routing metadata exposed to dashboards and integrations."""

    is_default: bool
    explicit_header: str = "x-computecop-endpoint"
    compatible_api_families: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EndpointRecord:
    """Full endpoint view combining route, capabilities, health, and routing."""

    name: str
    kind: EndpointKind
    base_url: str
    health_path: str
    timeout_seconds: float
    capabilities: EndpointCapabilities
    health: EndpointHealthStatus
    routing: EndpointRoutingMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "base_url": self.base_url,
            "health_path": self.health_path,
            "timeout_seconds": self.timeout_seconds,
            "capabilities": {
                "api_family": self.capabilities.api_family.value,
                "supports_streaming": self.capabilities.supports_streaming,
                "supports_model_list": self.capabilities.supports_model_list,
                "supports_offload": self.capabilities.supports_offload,
                "default_context_tokens": self.capabilities.default_context_tokens,
                "default_output_tokens": self.capabilities.default_output_tokens,
            },
            "health": {
                "healthy": self.health.healthy,
                "status_code": self.health.status_code,
                "latency_ms": self.health.latency_ms,
                "failure_rate": self.health.failure_rate,
                "failure_streak": self.health.failure_streak,
                "last_success_at": (
                    self.health.last_success_at.isoformat()
                    if self.health.last_success_at is not None
                    else None
                ),
                "checked_at": (
                    self.health.checked_at.isoformat()
                    if self.health.checked_at is not None
                    else None
                ),
                "detail": self.health.detail,
                "status_category": self.health.status_category,
                "stale": self.health.stale,
                "circuit_breaker": (
                    self.health.circuit_breaker.to_dict()
                    if self.health.circuit_breaker is not None
                    else None
                ),
            },
            "routing": {
                "is_default": self.routing.is_default,
                "explicit_header": self.routing.explicit_header,
                "compatible_api_families": list(self.routing.compatible_api_families),
            },
        }


@dataclass(slots=True)
class _ProbeCacheEntry:
    probe: HealthProbe
    cached_at_monotonic: float
    total_probes: int = 0
    failed_probes: int = 0


class EndpointCapabilityRegistry:
    """Probe, cache, and expose endpoint capabilities for routing decisions."""

    def __init__(
        self,
        router: UpstreamRouter,
        *,
        probe_ttl_seconds: float = 30.0,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
        half_open_successes: int = 1,
    ) -> None:
        if probe_ttl_seconds <= 0:
            raise ValueError("probe_ttl_seconds must be positive")
        self._router = router
        self._probe_ttl_seconds = probe_ttl_seconds
        self._cache: dict[str, _ProbeCacheEntry] = {}
        self._circuit_breakers = CircuitBreakerRegistry(
            list(router.routes.keys()),
            failure_threshold=failure_threshold,
            cooldown_seconds=cooldown_seconds,
            half_open_successes=half_open_successes,
        )

    @property
    def probe_ttl_seconds(self) -> float:
        return self._probe_ttl_seconds

    @property
    def circuit_breakers(self) -> CircuitBreakerRegistry:
        return self._circuit_breakers

    def allows_traffic(self, endpoint_name: str) -> bool:
        """Return whether routing may send traffic to the endpoint."""

        return self._circuit_breakers.allows_traffic(endpoint_name)

    def record_upstream_success(self, endpoint_name: str) -> CircuitBreakerStatus:
        """Record a successful upstream request for circuit breaker state."""

        return self._circuit_breakers.record_success(endpoint_name)

    def record_upstream_failure(self, endpoint_name: str) -> CircuitBreakerStatus:
        """Record a failed upstream request for circuit breaker state."""

        return self._circuit_breakers.record_failure(endpoint_name)

    def capabilities_for(self, route: EndpointRoute) -> EndpointCapabilities:
        """Return static capabilities derived from endpoint kind and config."""

        return EndpointCapabilities(
            api_family=route.kind,
            supports_streaming=route.supports_streaming,
            supports_model_list=_supports_model_list(route.kind),
            supports_offload=_supports_offload(route.kind),
            default_context_tokens=_DEFAULT_CONTEXT_TOKENS[route.kind],
            default_output_tokens=_DEFAULT_OUTPUT_TOKENS[route.kind],
        )

    async def probe(
        self,
        route: EndpointRoute,
        *,
        force: bool = False,
    ) -> EndpointHealthStatus:
        """Return a health snapshot, probing upstream when the cache is stale."""

        cached = self._cache.get(route.name)
        now = time.monotonic()
        if (
            not force
            and cached is not None
            and (now - cached.cached_at_monotonic) < self._probe_ttl_seconds
        ):
            return _health_from_probe(
                cached.probe,
                failure_rate=_failure_rate(cached),
                stale=False,
                circuit_breaker=self._circuit_breakers.snapshot(route.name),
            )

        probe = await self._router.probe(route)
        entry = cached or _ProbeCacheEntry(
            probe=probe,
            cached_at_monotonic=now,
            total_probes=0,
            failed_probes=0,
        )
        entry.total_probes += 1
        if probe.healthy:
            self._circuit_breakers.record_success(route.name)
        else:
            entry.failed_probes += 1
            self._circuit_breakers.record_failure(route.name)
        entry.probe = probe
        entry.cached_at_monotonic = now
        self._cache[route.name] = entry
        return _health_from_probe(
            probe,
            failure_rate=_failure_rate(entry),
            stale=False,
            circuit_breaker=self._circuit_breakers.snapshot(route.name),
        )

    async def record(
        self,
        route: EndpointRoute,
        *,
        force_probe: bool = False,
    ) -> EndpointRecord:
        """Build a full endpoint record with capabilities and health."""

        health = await self.probe(route, force=force_probe)
        return EndpointRecord(
            name=route.name,
            kind=route.kind,
            base_url=route.base_url,
            health_path=route.health_path,
            timeout_seconds=route.timeout_seconds,
            capabilities=self.capabilities_for(route),
            health=health,
            routing=EndpointRoutingMetadata(
                is_default=route.name == self._router.default_route.name,
                compatible_api_families=_compatible_families(route.kind),
            ),
        )

    async def list_records(self, *, force_probe: bool = False) -> list[EndpointRecord]:
        """Return endpoint records for every configured route."""

        records: list[EndpointRecord] = []
        for route in self._router.routes.values():
            records.append(await self.record(route, force_probe=force_probe))
        return sorted(records, key=lambda item: item.name)

    def select_compatible(
        self,
        *,
        family: str | EndpointKind | None = None,
        requires_streaming: bool = False,
        prefer_healthy: bool = True,
    ) -> EndpointRoute | None:
        """Choose the best configured endpoint for an API family."""

        target_kind = _resolve_family(family)
        if target_kind is None:
            return self._router.default_route

        candidates = [
            route
            for route in self._router.routes.values()
            if route.kind == target_kind and _route_supports_streaming(route, requires_streaming)
        ]
        if not candidates:
            return None

        if not prefer_healthy:
            return candidates[0]

        available = [
            route for route in candidates if self._circuit_breakers.allows_traffic(route.name)
        ]
        if not available:
            return None

        healthy = [route for route in available if _cached_is_healthy(self._cache.get(route.name))]
        pool = healthy or available
        return _select_lowest_failure_rate(pool, self._cache)

    def invalidate(self, name: str | None = None) -> None:
        """Drop cached probe data for one endpoint or all endpoints."""

        if name is None:
            self._cache.clear()
            return
        self._cache.pop(name, None)


def resolve_api_family(family: str | EndpointKind | None) -> EndpointKind | None:
    """Normalize a route family hint to an endpoint kind."""

    return _resolve_family(family)


def _resolve_family(family: str | EndpointKind | None) -> EndpointKind | None:
    if family is None:
        return None
    if isinstance(family, EndpointKind):
        return family
    normalized = family.strip().lower()
    if normalized in _FAMILY_ALIASES:
        return _FAMILY_ALIASES[normalized]
    try:
        return EndpointKind(normalized)
    except ValueError:
        return None


def _supports_model_list(kind: EndpointKind) -> bool:
    return kind in {EndpointKind.OLLAMA, EndpointKind.OPENAI_COMPATIBLE}


def _supports_offload(kind: EndpointKind) -> bool:
    return kind in {EndpointKind.OLLAMA, EndpointKind.LLAMA_CPP}


def _compatible_families(kind: EndpointKind) -> tuple[str, ...]:
    if kind == EndpointKind.OLLAMA:
        return ("ollama",)
    if kind == EndpointKind.LLAMA_CPP:
        return ("llama_cpp",)
    return ("openai", "openai_compatible")


def _route_supports_streaming(route: EndpointRoute, requires_streaming: bool) -> bool:
    return route.supports_streaming if requires_streaming else True


def _failure_rate(entry: _ProbeCacheEntry) -> float:
    if entry.total_probes <= 0:
        return 0.0
    return round(entry.failed_probes / entry.total_probes, 4)


def _cached_is_healthy(entry: _ProbeCacheEntry | None) -> bool:
    if entry is None:
        return True
    return entry.probe.healthy


def _select_lowest_failure_rate(
    routes: list[EndpointRoute],
    cache: dict[str, _ProbeCacheEntry],
) -> EndpointRoute:
    def sort_key(route: EndpointRoute) -> tuple[float, int, str]:
        entry = cache.get(route.name)
        failure_rate = _failure_rate(entry) if entry is not None else 0.0
        streak = entry.probe.failure_streak if entry is not None else 0
        return (failure_rate, streak, route.name)

    return sorted(routes, key=sort_key)[0]


def _health_from_probe(
    probe: HealthProbe,
    *,
    failure_rate: float,
    stale: bool,
    circuit_breaker: CircuitBreakerStatus | None = None,
) -> EndpointHealthStatus:
    status_category = probe.failure_category.value if probe.failure_category is not None else None
    return EndpointHealthStatus(
        healthy=probe.healthy,
        status_code=probe.status_code,
        latency_ms=probe.latency_ms,
        failure_rate=failure_rate,
        failure_streak=probe.failure_streak,
        last_success_at=probe.last_success_at,
        checked_at=probe.checked_at or utc_now(),
        detail=probe.detail,
        status_category=status_category,
        stale=stale,
        circuit_breaker=circuit_breaker,
    )
