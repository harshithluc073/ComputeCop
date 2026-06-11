from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from computecop.app import create_app
from computecop.config import EndpointConfig, RuntimeConfig
from computecop.endpoints import EndpointCapabilityRegistry
from computecop.health import (
    CircuitBreakerRegistry,
    CircuitBreakerState,
    EndpointCircuitBreaker,
    EndpointHealthWatcher,
)
from computecop.models import EndpointKind, EndpointRoute
from computecop.upstream import UpstreamRouter


def _ollama_route() -> EndpointRoute:
    return EndpointRoute(
        name="ollama",
        kind=EndpointKind.OLLAMA,
        base_url="http://ollama.test",
        timeout_seconds=30.0,
        health_path="/api/tags",
        supports_streaming=True,
    )


def _openai_route() -> EndpointRoute:
    return EndpointRoute(
        name="openai",
        kind=EndpointKind.OPENAI_COMPATIBLE,
        base_url="http://openai.test",
        timeout_seconds=30.0,
        health_path="/v1/models",
        supports_streaming=True,
    )


def test_circuit_breaker_opens_after_failure_threshold() -> None:
    breaker = EndpointCircuitBreaker(failure_threshold=3, cooldown_seconds=30.0)

    breaker.record_failure()
    assert breaker.snapshot().state is CircuitBreakerState.CLOSED
    breaker.record_failure()
    assert breaker.snapshot().state is CircuitBreakerState.CLOSED

    opened = breaker.record_failure()
    assert opened.state is CircuitBreakerState.OPEN
    assert opened.consecutive_failures == 3
    assert breaker.allows_traffic() is False


def test_circuit_breaker_recovers_through_half_open() -> None:
    breaker = EndpointCircuitBreaker(failure_threshold=2, cooldown_seconds=0.05)

    breaker.record_failure()
    breaker.record_failure()
    assert breaker.snapshot().state is CircuitBreakerState.OPEN

    time.sleep(0.12)
    assert breaker.snapshot().state is CircuitBreakerState.HALF_OPEN
    assert breaker.allows_traffic() is True

    recovered = breaker.record_success()
    assert recovered.state is CircuitBreakerState.CLOSED
    assert recovered.consecutive_failures == 0
    assert breaker.allows_traffic() is True


def test_circuit_breaker_half_open_failure_reopens() -> None:
    breaker = EndpointCircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)

    breaker.record_failure()
    assert breaker.snapshot().state is CircuitBreakerState.OPEN

    time.sleep(0.12)
    assert breaker.snapshot().state is CircuitBreakerState.HALF_OPEN

    reopened = breaker.record_failure()
    assert reopened.state is CircuitBreakerState.OPEN
    assert breaker.allows_traffic() is False


def test_circuit_breaker_registry_tracks_endpoints_independently() -> None:
    registry = CircuitBreakerRegistry(
        ["alpha", "beta"],
        failure_threshold=2,
        cooldown_seconds=30.0,
    )

    registry.record_failure("alpha")
    registry.record_failure("alpha")

    assert registry.snapshot("alpha").state is CircuitBreakerState.OPEN
    assert registry.snapshot("beta").state is CircuitBreakerState.CLOSED
    assert registry.allows_traffic("alpha") is False
    assert registry.allows_traffic("beta") is True


@pytest.mark.asyncio
async def test_probe_updates_circuit_breaker_state() -> None:
    healthy = True

    def handler(request: httpx.Request) -> httpx.Response:
        if healthy:
            return httpx.Response(200)
        raise httpx.ConnectError("refused", request=request)

    router = UpstreamRouter([_ollama_route()])
    await router._client.aclose()
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    registry = EndpointCapabilityRegistry(
        router,
        probe_ttl_seconds=30.0,
        failure_threshold=2,
        cooldown_seconds=30.0,
    )
    try:
        ok = await registry.probe(_ollama_route(), force=True)
        assert ok.circuit_breaker is not None
        assert ok.circuit_breaker.state is CircuitBreakerState.CLOSED

        healthy = False
        await registry.probe(_ollama_route(), force=True)
        opened = await registry.probe(_ollama_route(), force=True)
        assert opened.circuit_breaker is not None
        assert opened.circuit_breaker.state is CircuitBreakerState.OPEN
    finally:
        await router.close()


@pytest.mark.asyncio
async def test_select_compatible_skips_open_circuit_breakers() -> None:
    router = UpstreamRouter([_ollama_route(), _openai_route()])
    registry = EndpointCapabilityRegistry(
        router,
        probe_ttl_seconds=30.0,
        failure_threshold=1,
        cooldown_seconds=30.0,
    )
    registry.record_upstream_failure("ollama")

    selected = registry.select_compatible(family="ollama")
    assert selected is None

    selected_openai = registry.select_compatible(family="openai")
    assert selected_openai is not None
    assert selected_openai.name == "openai"


@pytest.mark.asyncio
async def test_health_watcher_probes_all_endpoints() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host or "")
        return httpx.Response(200)

    router = UpstreamRouter([_ollama_route(), _openai_route()])
    await router._client.aclose()
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    registry = EndpointCapabilityRegistry(router, probe_ttl_seconds=30.0)
    watcher = EndpointHealthWatcher(registry, router, interval_seconds=0.05, jitter_fraction=0.0)
    try:
        await watcher.start()
        await asyncio.sleep(0.15)
        await watcher.stop()
    finally:
        await router.close()

    assert "ollama.test" in calls
    assert "openai.test" in calls


@pytest.mark.asyncio
async def test_breaker_recovery_allows_routing_again() -> None:
    router = UpstreamRouter([_ollama_route()])
    registry = EndpointCapabilityRegistry(
        router,
        probe_ttl_seconds=0.01,
        failure_threshold=1,
        cooldown_seconds=0.05,
    )

    healthy = False

    def handler(request: httpx.Request) -> httpx.Response:
        if healthy:
            return httpx.Response(200)
        raise httpx.ConnectError("refused", request=request)

    await router._client.aclose()
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await registry.probe(_ollama_route(), force=True)
        assert registry.select_compatible(family="ollama") is None

        healthy = True
        await asyncio.sleep(0.06)
        recovered = await registry.probe(_ollama_route(), force=True)
        assert recovered.circuit_breaker is not None
        assert recovered.circuit_breaker.state is CircuitBreakerState.CLOSED
        assert registry.select_compatible(family="ollama") is not None
    finally:
        await router.close()


@pytest.mark.asyncio
async def test_endpoints_api_exposes_circuit_breaker_status(tmp_path) -> None:
    app = create_app(
        RuntimeConfig(
            event_log_path=tmp_path / "events.jsonl",
            endpoint_registry={
                "capability_probe_ttl_seconds": 30.0,
                "health_watcher_enabled": False,
                "circuit_breaker_failure_threshold": 1,
                "circuit_breaker_cooldown_seconds": 30.0,
            },
            endpoints=[
                EndpointConfig(
                    name="ollama",
                    kind=EndpointKind.OLLAMA,
                    base_url="http://ollama.test",
                    health_path="/api/tags",
                )
            ],
        )
    )
    router = app.state.runtime.upstream

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    await router._client.aclose()
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/endpoints?refresh=true")

    assert response.status_code == 200
    endpoint = response.json()["endpoints"][0]
    breaker = endpoint["health"]["circuit_breaker"]
    assert breaker is not None
    assert breaker["state"] == "open"
    assert breaker["allows_traffic"] is False
