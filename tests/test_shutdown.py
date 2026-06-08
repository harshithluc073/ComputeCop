from __future__ import annotations

import asyncio

import pytest

from computecop.app import build_runtime
from computecop.config import RuntimeConfig
from computecop.models import EndpointKind, EndpointRoute
from computecop.shutdown import ShutdownCoordinator
from computecop.upstream import UpstreamRouter


@pytest.mark.asyncio
async def test_shutdown_coordinator_is_idempotent() -> None:
    runtime = build_runtime(RuntimeConfig())
    coordinator = ShutdownCoordinator()
    await runtime.start()
    await coordinator.shutdown_runtime(runtime, drain_timeout_seconds=0)
    await coordinator.shutdown_runtime(runtime, drain_timeout_seconds=0)
    assert coordinator.shutdown_complete is True
    await runtime.upstream.close()


@pytest.mark.asyncio
async def test_shutdown_request_is_idempotent() -> None:
    coordinator = ShutdownCoordinator()
    assert coordinator.request_shutdown() is True
    assert coordinator.request_shutdown() is False


@pytest.mark.asyncio
async def test_upstream_close_is_idempotent() -> None:
    router = UpstreamRouter(
        [
            EndpointRoute(
                name="local",
                kind=EndpointKind.OPENAI_COMPATIBLE,
                base_url="http://127.0.0.1:8080",
                timeout_seconds=1.0,
                health_path="/health",
            )
        ]
    )
    await router.close()
    await router.close()