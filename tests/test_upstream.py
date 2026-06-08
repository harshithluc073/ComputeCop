from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from computecop.app import create_app
from computecop.config import EndpointConfig, RuntimeConfig
from computecop.models import EndpointKind, EndpointRoute
from computecop.upstream import (
    UpstreamFailure,
    UpstreamFailureCategory,
    UpstreamRouter,
)

_REQUEST = httpx.Request("POST", "http://endpoint.test/v1/chat/completions")


def _route() -> EndpointRoute:
    return EndpointRoute(
        name="local",
        kind=EndpointKind.OPENAI_COMPATIBLE,
        base_url="http://endpoint.test",
        timeout_seconds=12.0,
        health_path="/health",
    )


def _classify(exc: Exception, *, streaming: bool = False) -> UpstreamFailure:
    return UpstreamFailure.from_transport_error(
        exc,
        endpoint="local",
        base_url="http://endpoint.test",
        timeout_seconds=12.0,
        streaming=streaming,
    )


@pytest.mark.parametrize(
    ("exc", "category", "status_code", "retryable"),
    [
        (
            httpx.ConnectError("refused", request=_REQUEST),
            UpstreamFailureCategory.UNREACHABLE,
            502,
            True,
        ),
        (
            httpx.ConnectTimeout("connect timed out", request=_REQUEST),
            UpstreamFailureCategory.UNREACHABLE,
            502,
            True,
        ),
        (
            httpx.ReadTimeout("read timed out", request=_REQUEST),
            UpstreamFailureCategory.TIMEOUT,
            504,
            True,
        ),
        (
            httpx.UnsupportedProtocol("bad scheme", request=_REQUEST),
            UpstreamFailureCategory.MISCONFIGURED_ENDPOINT,
            502,
            False,
        ),
        (
            httpx.InvalidURL("bad url"),
            UpstreamFailureCategory.MISCONFIGURED_ENDPOINT,
            502,
            False,
        ),
        (
            httpx.DecodingError("bad bytes", request=_REQUEST),
            UpstreamFailureCategory.INVALID_RESPONSE,
            502,
            False,
        ),
    ],
)
def test_transport_errors_map_to_categories(
    exc: Exception,
    category: UpstreamFailureCategory,
    status_code: int,
    retryable: bool,
) -> None:
    failure = _classify(exc)
    assert failure.category is category
    assert failure.status_code == status_code
    assert failure.retryable is retryable
    assert failure.endpoint == "local"
    assert failure.remediation


def test_status_error_maps_status_and_retryability() -> None:
    response = httpx.Response(503, request=_REQUEST)
    failure = _classify(httpx.HTTPStatusError("bad", request=_REQUEST, response=response))
    assert failure.category is UpstreamFailureCategory.STATUS_ERROR
    assert failure.status_code == 503
    assert failure.retryable is True

    response_400 = httpx.Response(400, request=_REQUEST)
    non_retryable = _classify(httpx.HTTPStatusError("bad", request=_REQUEST, response=response_400))
    assert non_retryable.category is UpstreamFailureCategory.STATUS_ERROR
    assert non_retryable.status_code == 400
    assert non_retryable.retryable is False


def test_remote_protocol_error_distinguishes_stream_vs_buffered() -> None:
    streamed = _classify(httpx.RemoteProtocolError("cut", request=_REQUEST), streaming=True)
    assert streamed.category is UpstreamFailureCategory.STREAM_INTERRUPTED
    assert streamed.retryable is True

    buffered = _classify(httpx.RemoteProtocolError("cut", request=_REQUEST), streaming=False)
    assert buffered.category is UpstreamFailureCategory.INVALID_RESPONSE


def test_unknown_route_raises_route_not_found() -> None:
    router = UpstreamRouter([_route()])
    with pytest.raises(UpstreamFailure) as excinfo:
        router.route("missing")
    failure = excinfo.value
    assert failure.category is UpstreamFailureCategory.ROUTE_NOT_FOUND
    assert failure.status_code == 400
    assert "local" in (failure.remediation or "")


@pytest.mark.asyncio
async def test_router_request_classifies_connect_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    router = UpstreamRouter([_route()])
    await router._client.aclose()
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(UpstreamFailure) as excinfo:
            await router.request(
                _route(),
                method="POST",
                path="/v1/chat/completions",
                headers={},
                json_body={"model": "local"},
            )
    finally:
        await router.close()
    assert excinfo.value.category is UpstreamFailureCategory.UNREACHABLE


@pytest.mark.asyncio
async def test_router_stream_classifies_interruption() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.RemoteProtocolError("closed early", request=request)

    router = UpstreamRouter([_route()])
    await router._client.aclose()
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(UpstreamFailure) as excinfo:
            async for _ in router.stream(
                _route(),
                method="POST",
                path="/v1/chat/completions",
                headers={},
                json_body={"model": "local", "stream": True},
            ):
                pass
    finally:
        await router.close()
    assert excinfo.value.category is UpstreamFailureCategory.STREAM_INTERRUPTED


@pytest.mark.asyncio
async def test_proxy_returns_typed_upstream_failure(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.state.runtime.upstream = _FailingUpstream(
        UpstreamFailure(
            "endpoint 'ollama' is unreachable at http://ollama.test",
            category=UpstreamFailureCategory.UNREACHABLE,
            status_code=502,
            endpoint="ollama",
            retryable=True,
            remediation="start the local inference engine",
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"x-computecop-class": "prompt"},
            json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
        )
        events = await client.get("/events")

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["type"] == "computecop_upstream_unreachable"
    assert body["upstream_failure"]["category"] == "unreachable"
    assert body["upstream_failure"]["retryable"] is True
    assert response.headers["retry-after"]
    kinds = {event["kind"] for event in events.json()["events"]}
    assert "upstream.failure" in kinds


@pytest.mark.asyncio
async def test_proxy_route_not_found_failure(tmp_path: Path) -> None:
    app = _app(tmp_path)
    app.state.runtime.upstream = _FailingUpstream(
        UpstreamFailure(
            "unknown upstream endpoint 'nope'",
            category=UpstreamFailureCategory.ROUTE_NOT_FOUND,
            status_code=400,
            endpoint="nope",
            retryable=False,
            remediation="route to a configured endpoint",
        ),
        raise_on_route=True,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"x-computecop-class": "prompt", "x-computecop-endpoint": "nope"},
            json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["type"] == "computecop_upstream_route_not_found"
    assert "retry-after" not in response.headers


class _FailingUpstream:
    def __init__(self, failure: UpstreamFailure, *, raise_on_route: bool = False) -> None:
        self.failure = failure
        self.raise_on_route = raise_on_route
        self.routes = {
            "ollama": EndpointRoute(
                name="ollama",
                kind=EndpointKind.OLLAMA,
                base_url="http://ollama.test",
                timeout_seconds=30.0,
                health_path="/api/tags",
            )
        }

    def route(self, name: str | None = None) -> EndpointRoute:
        if self.raise_on_route:
            raise self.failure
        return self.routes["ollama"]

    async def request(self, route, *, method, path, headers, json_body, content=None):
        raise self.failure

    async def stream(self, route, *, method, path, headers, json_body):
        raise self.failure
        yield b""  # pragma: no cover - generator marker

    async def close(self) -> None:
        return None


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


def test_failure_to_dict_is_json_safe() -> None:
    failure = UpstreamFailure(
        "boom",
        category=UpstreamFailureCategory.TIMEOUT,
        status_code=504,
        endpoint="local",
        retryable=True,
        remediation="retry",
    )
    payload: dict[str, Any] = failure.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["category"] == "timeout"


@pytest.mark.asyncio
async def test_probe_records_latency_and_resets_failures_on_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    router = UpstreamRouter([_route()])
    await router._client.aclose()
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        probe = await router.probe(_route())
    finally:
        await router.close()

    assert probe.healthy is True
    assert probe.status_code == 200
    assert probe.failure_streak == 0
    assert probe.latency_ms is not None and probe.latency_ms >= 0.0
    assert probe.last_success_at is not None
    assert probe.checked_at is not None
    assert probe.base_url == "http://endpoint.test"
    assert probe.health_path == "/health"
    assert probe.failure_category is None


@pytest.mark.asyncio
async def test_probe_tracks_failure_streak_and_category() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    router = UpstreamRouter([_route()])
    await router._client.aclose()
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        first = await router.probe(_route())
        second = await router.probe(_route())
    finally:
        await router.close()

    assert first.healthy is False
    assert first.failure_category is UpstreamFailureCategory.UNREACHABLE
    assert first.failure_streak == 1
    assert second.failure_streak == 2
    assert second.last_success_at is None


@pytest.mark.asyncio
async def test_probe_marks_server_error_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    router = UpstreamRouter([_route()])
    await router._client.aclose()
    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        probe = await router.probe(_route())
    finally:
        await router.close()

    assert probe.healthy is False
    assert probe.status_code == 503
    assert probe.failure_category is UpstreamFailureCategory.STATUS_ERROR
    assert probe.failure_streak == 1
