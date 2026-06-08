"""HTTP routing to local inference endpoints."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

from computecop.models import EndpointRoute, utc_now

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class UpstreamFailureCategory(str, Enum):
    """Actionable category for an upstream endpoint failure."""

    UNREACHABLE = "unreachable"
    TIMEOUT = "timeout"
    ROUTE_NOT_FOUND = "route_not_found"
    STATUS_ERROR = "status_error"
    STREAM_INTERRUPTED = "stream_interrupted"
    INVALID_RESPONSE = "invalid_response"
    MISCONFIGURED_ENDPOINT = "misconfigured_endpoint"


class UpstreamFailure(RuntimeError):
    """Structured upstream endpoint failure with diagnostics and remediation."""

    def __init__(
        self,
        message: str,
        *,
        category: UpstreamFailureCategory,
        status_code: int = 502,
        endpoint: str | None = None,
        retryable: bool = False,
        remediation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.category = category
        self.status_code = status_code
        self.endpoint = endpoint
        self.retryable = retryable
        self.remediation = remediation

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of the failure."""

        return {
            "message": self.message,
            "category": self.category.value,
            "status_code": self.status_code,
            "endpoint": self.endpoint,
            "retryable": self.retryable,
            "remediation": self.remediation,
        }

    @classmethod
    def from_transport_error(
        cls,
        exc: Exception,
        *,
        endpoint: str,
        base_url: str,
        timeout_seconds: float,
        streaming: bool = False,
    ) -> UpstreamFailure:
        """Classify an httpx transport error into a typed upstream failure."""

        return _classify_transport_error(
            exc,
            endpoint=endpoint,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            streaming=streaming,
        )


# Backwards-compatible alias retained for callers that imported the previous name.
UpstreamError = UpstreamFailure


@dataclass(frozen=True, slots=True)
class UpstreamResponse:
    """Buffered upstream response."""

    status_code: int
    headers: dict[str, str]
    content: bytes


@dataclass(frozen=True, slots=True)
class HealthProbe:
    """Endpoint health probe result with latency and failure history."""

    endpoint: str
    healthy: bool
    status_code: int | None
    detail: str
    base_url: str = ""
    health_path: str = ""
    latency_ms: float | None = None
    failure_category: UpstreamFailureCategory | None = None
    failure_streak: int = 0
    last_success_at: datetime | None = None
    checked_at: datetime | None = None


@dataclass(slots=True)
class _EndpointHealth:
    """Mutable per-endpoint health bookkeeping for probe diagnostics."""

    failure_streak: int = 0
    last_success_at: datetime | None = None


class UpstreamRouter:
    """Route proxy requests to configured local inference endpoints."""

    def __init__(self, routes: list[EndpointRoute]) -> None:
        if not routes:
            raise ValueError("at least one upstream route is required")
        self.routes = {route.name: route for route in routes}
        self.default_route = routes[0]
        self._client = httpx.AsyncClient(follow_redirects=False)
        self._health: dict[str, _EndpointHealth] = {
            route.name: _EndpointHealth() for route in routes
        }

    async def close(self) -> None:
        await self._client.aclose()

    def route(self, name: str | None = None) -> EndpointRoute:
        if name is None:
            return self.default_route
        try:
            return self.routes[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.routes))
            raise UpstreamFailure(
                f"unknown upstream endpoint '{name}', configured endpoints: {known}",
                category=UpstreamFailureCategory.ROUTE_NOT_FOUND,
                status_code=400,
                endpoint=name,
                retryable=False,
                remediation=(
                    f"route to one of the configured endpoints ({known}) "
                    "or correct the x-computecop-endpoint header"
                ),
            ) from exc

    async def probe(self, route: EndpointRoute | None = None) -> HealthProbe:
        target = route or self.default_route
        history = self._health.setdefault(target.name, _EndpointHealth())
        started = time.perf_counter()
        try:
            response = await self._client.get(
                f"{target.base_url}{target.health_path}",
                timeout=min(10.0, target.timeout_seconds),
            )
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            failure = _classify_transport_error(
                exc,
                endpoint=target.name,
                base_url=target.base_url,
                timeout_seconds=target.timeout_seconds,
                streaming=False,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            history.failure_streak += 1
            return HealthProbe(
                endpoint=target.name,
                healthy=False,
                status_code=None,
                detail=failure.message,
                base_url=target.base_url,
                health_path=target.health_path,
                latency_ms=latency_ms,
                failure_category=failure.category,
                failure_streak=history.failure_streak,
                last_success_at=history.last_success_at,
                checked_at=utc_now(),
            )

        latency_ms = (time.perf_counter() - started) * 1000.0
        healthy = response.status_code < 500
        checked_at = utc_now()
        if healthy:
            history.failure_streak = 0
            history.last_success_at = checked_at
            failure_category = None
        else:
            history.failure_streak += 1
            failure_category = UpstreamFailureCategory.STATUS_ERROR
        return HealthProbe(
            endpoint=target.name,
            healthy=healthy,
            status_code=response.status_code,
            detail=response.reason_phrase or ("OK" if healthy else "server error"),
            base_url=target.base_url,
            health_path=target.health_path,
            latency_ms=latency_ms,
            failure_category=failure_category,
            failure_streak=history.failure_streak,
            last_success_at=history.last_success_at,
            checked_at=checked_at,
        )

    async def request(
        self,
        route: EndpointRoute,
        *,
        method: str,
        path: str,
        headers: Mapping[str, str],
        json_body: Any | None,
        content: bytes | None = None,
    ) -> UpstreamResponse:
        """Forward a buffered request to an upstream endpoint."""

        try:
            response = await self._client.request(
                method=method,
                url=f"{route.base_url}{path}",
                headers=_forward_headers(headers),
                json=json_body,
                content=content,
                timeout=route.timeout_seconds,
            )
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise _classify_transport_error(
                exc,
                endpoint=route.name,
                base_url=route.base_url,
                timeout_seconds=route.timeout_seconds,
                streaming=False,
            ) from exc

        return UpstreamResponse(
            status_code=response.status_code,
            headers=_response_headers(response.headers),
            content=response.content,
        )

    async def stream(
        self,
        route: EndpointRoute,
        *,
        method: str,
        path: str,
        headers: Mapping[str, str],
        json_body: Any | None,
    ) -> AsyncIterator[bytes]:
        """Forward a streaming request and yield bytes from the upstream response."""

        try:
            async with self._client.stream(
                method=method,
                url=f"{route.base_url}{path}",
                headers=_forward_headers(headers),
                json=json_body,
                timeout=route.timeout_seconds,
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise _classify_transport_error(
                exc,
                endpoint=route.name,
                base_url=route.base_url,
                timeout_seconds=route.timeout_seconds,
                streaming=True,
            ) from exc


def _classify_transport_error(
    exc: Exception,
    *,
    endpoint: str,
    base_url: str,
    timeout_seconds: float,
    streaming: bool,
) -> UpstreamFailure:
    """Map an httpx transport error onto the ComputeCop upstream failure taxonomy."""

    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        return UpstreamFailure(
            f"endpoint '{endpoint}' returned HTTP {status_code}",
            category=UpstreamFailureCategory.STATUS_ERROR,
            status_code=status_code,
            endpoint=endpoint,
            retryable=status_code in RETRYABLE_STATUS_CODES,
            remediation=(f"inspect the {endpoint} server logs for the HTTP {status_code} response"),
        )

    if isinstance(exc, (httpx.InvalidURL, httpx.UnsupportedProtocol)):
        return UpstreamFailure(
            f"endpoint '{endpoint}' is misconfigured: {exc!r}",
            category=UpstreamFailureCategory.MISCONFIGURED_ENDPOINT,
            status_code=502,
            endpoint=endpoint,
            retryable=False,
            remediation=(f"check the configured base_url '{base_url}' for endpoint '{endpoint}'"),
        )

    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return UpstreamFailure(
            f"endpoint '{endpoint}' is unreachable at {base_url}",
            category=UpstreamFailureCategory.UNREACHABLE,
            status_code=502,
            endpoint=endpoint,
            retryable=True,
            remediation=(
                f"start the local inference engine and confirm it is listening at {base_url}"
            ),
        )

    if isinstance(exc, httpx.TimeoutException):
        return UpstreamFailure(
            f"endpoint '{endpoint}' timed out after {timeout_seconds:g}s",
            category=UpstreamFailureCategory.TIMEOUT,
            status_code=504,
            endpoint=endpoint,
            retryable=True,
            remediation=(
                "retry the request or raise the endpoint timeout if the model is slow to respond"
            ),
        )

    if isinstance(exc, httpx.RemoteProtocolError):
        if streaming:
            return UpstreamFailure(
                f"stream from endpoint '{endpoint}' was interrupted",
                category=UpstreamFailureCategory.STREAM_INTERRUPTED,
                status_code=502,
                endpoint=endpoint,
                retryable=True,
                remediation=(
                    "the endpoint closed the stream early; retry or check the server logs"
                ),
            )
        return UpstreamFailure(
            f"endpoint '{endpoint}' returned an invalid response: {exc!r}",
            category=UpstreamFailureCategory.INVALID_RESPONSE,
            status_code=502,
            endpoint=endpoint,
            retryable=True,
            remediation="the endpoint sent a malformed response; check the server version",
        )

    if isinstance(exc, httpx.DecodingError):
        return UpstreamFailure(
            f"endpoint '{endpoint}' returned an undecodable response: {exc!r}",
            category=UpstreamFailureCategory.INVALID_RESPONSE,
            status_code=502,
            endpoint=endpoint,
            retryable=False,
            remediation="the endpoint returned content ComputeCop could not decode",
        )

    if isinstance(exc, httpx.TransportError):
        return UpstreamFailure(
            f"endpoint '{endpoint}' transport error at {base_url}: {exc!r}",
            category=UpstreamFailureCategory.UNREACHABLE,
            status_code=502,
            endpoint=endpoint,
            retryable=True,
            remediation=(
                f"confirm the local inference engine is running and reachable at {base_url}"
            ),
        )

    return UpstreamFailure(
        f"endpoint '{endpoint}' returned an invalid response: {exc!r}",
        category=UpstreamFailureCategory.INVALID_RESPONSE,
        status_code=502,
        endpoint=endpoint,
        retryable=False,
        remediation="check the endpoint configuration and server logs",
    )


def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    blocked = {"host", "content-length", "connection", "accept-encoding"}
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


def _response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    blocked = {"content-length", "transfer-encoding", "connection", "content-encoding"}
    return {key: value for key, value in headers.items() if key.lower() not in blocked}
