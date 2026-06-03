"""HTTP routing to local inference endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from computecop.models import EndpointRoute


class UpstreamError(RuntimeError):
    """Raised when an upstream endpoint cannot be reached or returns invalid data."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class UpstreamResponse:
    """Buffered upstream response."""

    status_code: int
    headers: dict[str, str]
    content: bytes


@dataclass(frozen=True, slots=True)
class HealthProbe:
    """Endpoint health probe result."""

    endpoint: str
    healthy: bool
    status_code: int | None
    detail: str


class UpstreamRouter:
    """Route proxy requests to configured local inference endpoints."""

    def __init__(self, routes: list[EndpointRoute]) -> None:
        if not routes:
            raise ValueError("at least one upstream route is required")
        self.routes = {route.name: route for route in routes}
        self.default_route = routes[0]
        self._client = httpx.AsyncClient(follow_redirects=False)

    async def close(self) -> None:
        await self._client.aclose()

    def route(self, name: str | None = None) -> EndpointRoute:
        if name is None:
            return self.default_route
        try:
            return self.routes[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.routes))
            raise UpstreamError(f"unknown upstream endpoint '{name}', known: {known}", 400) from exc

    async def probe(self, route: EndpointRoute | None = None) -> HealthProbe:
        target = route or self.default_route
        try:
            response = await self._client.get(
                f"{target.base_url}{target.health_path}",
                timeout=min(10.0, target.timeout_seconds),
            )
            return HealthProbe(
                endpoint=target.name,
                healthy=response.status_code < 500,
                status_code=response.status_code,
                detail=response.reason_phrase,
            )
        except httpx.HTTPError as exc:
            return HealthProbe(target.name, False, None, repr(exc))

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
        except httpx.TimeoutException as exc:
            raise UpstreamError(f"upstream timeout from {route.name}", 504) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"upstream error from {route.name}: {exc!r}", 502) from exc

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
        except httpx.TimeoutException as exc:
            raise UpstreamError(f"upstream stream timeout from {route.name}", 504) from exc
        except httpx.HTTPStatusError as exc:
            raise UpstreamError(
                f"upstream stream returned {exc.response.status_code}",
                exc.response.status_code,
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"upstream stream error from {route.name}: {exc!r}", 502) from exc


def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    blocked = {"host", "content-length", "connection", "accept-encoding"}
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


def _response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    blocked = {"content-length", "transfer-encoding", "connection", "content-encoding"}
    return {key: value for key, value in headers.items() if key.lower() not in blocked}
