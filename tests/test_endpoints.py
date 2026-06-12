from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest

from computecop.app import create_app
from computecop.config import EndpointConfig, RuntimeConfig
from computecop.endpoints import EndpointCapabilityRegistry
from computecop.models import EndpointKind, EndpointRoute, RequestClass
from computecop.upstream import HealthProbe, UpstreamRouter


def _ollama_route() -> EndpointRoute:
    return EndpointRoute(
        name="ollama",
        kind=EndpointKind.OLLAMA,
        base_url="http://ollama.test",
        timeout_seconds=30.0,
        health_path="/api/tags",
        supports_streaming=True,
    )


def _llama_route() -> EndpointRoute:
    return EndpointRoute(
        name="llama-cpp",
        kind=EndpointKind.LLAMA_CPP,
        base_url="http://llama.test",
        timeout_seconds=30.0,
        health_path="/health",
        supports_streaming=True,
    )


def _openai_route() -> EndpointRoute:
    return EndpointRoute(
        name="openai",
        kind=EndpointKind.OPENAI_COMPATIBLE,
        base_url="http://openai.test",
        timeout_seconds=30.0,
        health_path="/v1/models",
        supports_streaming=False,
    )


@pytest.mark.asyncio
async def test_capabilities_derived_from_endpoint_kind() -> None:
    registry = EndpointCapabilityRegistry(UpstreamRouter([_ollama_route(), _llama_route()]))
    ollama_caps = registry.capabilities_for(_ollama_route())
    llama_caps = registry.capabilities_for(_llama_route())

    assert ollama_caps.api_family is EndpointKind.OLLAMA
    assert ollama_caps.supports_streaming is True
    assert ollama_caps.supports_model_list is True
    assert ollama_caps.supports_offload is True
    assert ollama_caps.default_context_tokens == 8192

    assert llama_caps.api_family is EndpointKind.LLAMA_CPP
    assert llama_caps.supports_model_list is False
    assert llama_caps.supports_offload is True


@pytest.mark.asyncio
async def test_probe_caches_results_until_ttl_expires() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200)

    router = UpstreamRouter([_ollama_route()])
    await router._client.aclose()
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    registry = EndpointCapabilityRegistry(router, probe_ttl_seconds=30.0)
    try:
        first = await registry.probe(_ollama_route())
        second = await registry.probe(_ollama_route())
        registry.invalidate("ollama")
        third = await registry.probe(_ollama_route(), force=True)
    finally:
        await router.close()

    assert first.healthy is True
    assert second.stale is False
    assert calls["count"] == 2
    assert third.healthy is True


@pytest.mark.asyncio
async def test_probe_tracks_failure_rate() -> None:
    healthy = True

    def handler(request: httpx.Request) -> httpx.Response:
        if healthy:
            return httpx.Response(200)
        raise httpx.ConnectError("refused", request=request)

    router = UpstreamRouter([_ollama_route()])
    await router._client.aclose()
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    registry = EndpointCapabilityRegistry(router, probe_ttl_seconds=0.01)
    try:
        ok = await registry.probe(_ollama_route(), force=True)
        healthy = False
        await asyncio.sleep(0.02)
        failed = await registry.probe(_ollama_route(), force=True)
    finally:
        await router.close()

    assert ok.failure_rate == 0.0
    assert failed.healthy is False
    assert failed.failure_rate == 0.5


@pytest.mark.asyncio
async def test_select_compatible_prefers_healthy_low_failure_endpoint() -> None:
    router = UpstreamRouter([_ollama_route(), _openai_route()])
    registry = EndpointCapabilityRegistry(router, probe_ttl_seconds=30.0)
    registry._cache["ollama"] = _cache_entry(healthy=False, failure_rate=0.8, streak=4)
    registry._cache["openai"] = _cache_entry(healthy=True, failure_rate=0.1, streak=0)

    selected = registry.select_compatible(family="openai")
    assert selected is not None
    assert selected.name == "openai"


@pytest.mark.asyncio
async def test_select_compatible_honors_streaming_requirement() -> None:
    router = UpstreamRouter([_openai_route(), _ollama_route()])
    registry = EndpointCapabilityRegistry(router, probe_ttl_seconds=30.0)

    assert registry.select_compatible(family="openai", requires_streaming=True) is None
    assert registry.select_compatible(family="ollama", requires_streaming=True) is not None


@pytest.mark.asyncio
async def test_list_records_includes_routing_metadata() -> None:
    router = UpstreamRouter([_ollama_route(), _llama_route()])

    async def fake_probe(route: EndpointRoute | None = None) -> HealthProbe:
        target = route or _ollama_route()
        return HealthProbe(
            endpoint=target.name,
            healthy=True,
            status_code=200,
            detail="OK",
            latency_ms=12.5,
            failure_streak=0,
        )

    router.probe = fake_probe  # type: ignore[method-assign]
    registry = EndpointCapabilityRegistry(router, probe_ttl_seconds=30.0)
    records = await registry.list_records(force_probe=True)

    assert [record.name for record in records] == ["llama-cpp", "ollama"]
    default = next(record for record in records if record.routing.is_default)
    assert default.name == "ollama"
    assert default.routing.compatible_api_families == ("ollama",)
    assert default.health.latency_ms == 12.5


@pytest.mark.asyncio
async def test_endpoints_api_returns_capability_records(tmp_path: Path) -> None:
    app = _app(tmp_path)
    router = app.state.runtime.upstream

    async def fake_probe(route: EndpointRoute | None = None) -> HealthProbe:
        target = route or router.default_route
        return HealthProbe(
            endpoint=target.name,
            healthy=True,
            status_code=200,
            detail="OK",
            latency_ms=4.2,
            failure_streak=0,
        )

    router.probe = fake_probe  # type: ignore[method-assign]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/endpoints")

    assert response.status_code == 200
    body = response.json()
    assert len(body["endpoints"]) == 1
    endpoint = body["endpoints"][0]
    assert endpoint["name"] == "ollama"
    assert endpoint["capabilities"]["supports_model_list"] is True
    assert endpoint["health"]["healthy"] is True
    assert endpoint["routing"]["explicit_header"] == "x-computecop-endpoint"


def _cache_entry(*, healthy: bool, failure_rate: float, streak: int):
    from computecop.endpoints import _ProbeCacheEntry

    total = 10
    failed = int(failure_rate * total)
    return _ProbeCacheEntry(
        probe=HealthProbe(
            endpoint="cached",
            healthy=healthy,
            status_code=200 if healthy else None,
            detail="OK" if healthy else "down",
            failure_streak=streak,
        ),
        cached_at_monotonic=time.monotonic(),
        total_probes=total,
        failed_probes=failed,
    )


def _app(tmp_path: Path):
    config = RuntimeConfig(
        event_log_path=tmp_path / "events.jsonl",
        endpoints=[
            EndpointConfig(
                name="ollama",
                kind=EndpointKind.OLLAMA,
                base_url="http://ollama.test",
                health_path="/api/tags",
            )
        ],
    )
    return create_app(config)


@pytest.mark.asyncio
async def test_select_compatible_with_model_compatibility() -> None:
    from computecop.residency import ModelResidencyTracker
    
    router = UpstreamRouter([_ollama_route(), _llama_route()])
    tracker = ModelResidencyTracker()
    registry = EndpointCapabilityRegistry(router, residency_tracker=tracker)
    
    # 1. No observations yet -> should treat both as compatible
    assert registry.select_compatible(family="ollama", model="llama3") is not None
    
    # 2. Add observation to ollama
    tracker.record_request("llama3:latest", "ollama", RequestClass.USER_PROMPT)
    selected = registry.select_compatible(family="llama_cpp", model="phi3")
    assert selected is not None
    assert selected.name == "llama-cpp"
    
    selected = registry.select_compatible(family="ollama", model="llama3")
    assert selected is not None
    assert selected.name == "ollama"

    assert registry.select_compatible(family="ollama", model="phi3") is None


@pytest.mark.asyncio
async def test_model_matches_helper() -> None:
    from computecop.residency import model_matches
    assert model_matches("llama3", "llama3:latest") is True
    assert model_matches("llama-3-8b", "models/llama3-8b.gguf") is True
    assert model_matches("phi_3", "phi-3") is True
    assert model_matches("gpt4", "gpt-3.5") is False


@pytest.mark.asyncio
async def test_precise_error_responses_when_no_compatible_endpoint(tmp_path: Path) -> None:
    config = RuntimeConfig(
        event_log_path=tmp_path / "events.jsonl",
        endpoints=[
            EndpointConfig(
                name="ollama",
                kind=EndpointKind.OLLAMA,
                base_url="http://ollama.test",
                health_path="/api/tags",
            )
        ],
    )
    app = create_app(config)
    
    # Open the circuit breaker for ollama so fallback is not allowed
    registry = app.state.runtime.endpoint_registry
    registry.record_upstream_failure("ollama")
    registry.record_upstream_failure("ollama")
    registry.record_upstream_failure("ollama")
    
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chat/completions", json={"model": "llama3"})
        assert response.status_code == 400
        assert "no configured endpoint supports API family 'openai'" in response.json()["error"]["message"]

        config_no_stream = RuntimeConfig(
            event_log_path=tmp_path / "events_no_stream.jsonl",
            endpoints=[
                EndpointConfig(
                    name="ollama",
                    kind=EndpointKind.OLLAMA,
                    base_url="http://ollama.test",
                    health_path="/api/tags",
                    supports_streaming=False,
                )
            ],
        )
        app_no_stream = create_app(config_no_stream)
        registry_ns = app_no_stream.state.runtime.endpoint_registry
        registry_ns.record_upstream_failure("ollama")
        registry_ns.record_upstream_failure("ollama")
        registry_ns.record_upstream_failure("ollama")
        
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_no_stream), base_url="http://test"
        ) as client_ns:
            response = await client_ns.post("/api/generate", json={"model": "llama3", "stream": True})
            assert response.status_code == 400
            assert "supports streaming requests" in response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_safe_endpoint_failover(tmp_path: Path) -> None:
    config = RuntimeConfig(
        event_log_path=tmp_path / "events.jsonl",
        endpoints=[
            EndpointConfig(
                name="ollama-1",
                kind=EndpointKind.OLLAMA,
                base_url="http://ollama-1.test",
                health_path="/api/tags",
            ),
            EndpointConfig(
                name="ollama-2",
                kind=EndpointKind.OLLAMA,
                base_url="http://ollama-2.test",
                health_path="/api/tags",
            )
        ],
    )
    app = create_app(config)
    router = app.state.runtime.upstream
    
    await router._client.aclose()
    
    calls = []
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if "ollama-1.test" in str(request.url):
            raise httpx.ConnectError("Connection refused", request=request)
        return httpx.Response(200, json={"done": True})
        
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/generate", json={"model": "llama3", "prompt": "test"})
        assert response.status_code == 200
        assert response.json()["done"] is True
        
    assert len(calls) == 2
    assert "ollama-1.test" in calls[0]
    assert "ollama-2.test" in calls[1]

