from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest

from computecop.app import create_app
from computecop.config import EndpointConfig, RuntimeConfig, ServerConfig
from computecop.models import EndpointKind


def _create_test_app(tmp_path: Path, host: str, enable_web_ui: bool) -> httpx.AsyncClient:
    config = RuntimeConfig(
        event_log_path=tmp_path / "events.jsonl",
        server=ServerConfig(
            host=host,
            enable_web_ui=enable_web_ui,
            expose_remote=True,
        ),
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
    return app


@pytest.mark.asyncio
async def test_local_bind_dashboard_access(tmp_path: Path) -> None:
    app = _create_test_app(tmp_path, host="127.0.0.1", enable_web_ui=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Dashboard index
        response = await client.get("/")
        assert response.status_code == 200
        assert "<title>ComputeCop Control Plane</title>" in response.text

        # API state
        state_response = await client.get("/state")
        assert state_response.status_code == 200


@pytest.mark.asyncio
async def test_non_local_bind_blocked_access(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    app = _create_test_app(tmp_path, host="0.0.0.0", enable_web_ui=False)

    with caplog.at_level(logging.WARNING):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Dashboard index is blocked
            response = await client.get("/")
            assert response.status_code == 403
            assert "Web UI is disabled on non-local binds" in response.text

            # State API is blocked
            state_response = await client.get("/state")
            assert state_response.status_code == 403
            assert state_response.json() == {
                "error": "Web UI is disabled on non-local binds unless explicitly enabled"
            }

            # Decisions details are blocked
            decisions_response = await client.get("/decisions/some-id")
            assert decisions_response.status_code == 403

        # Assert warning was logged
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Blocked Web UI request" in w for w in warnings)


@pytest.mark.asyncio
async def test_non_local_bind_allowed_access(tmp_path: Path) -> None:
    app = _create_test_app(tmp_path, host="0.0.0.0", enable_web_ui=True)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Dashboard index is allowed
        response = await client.get("/")
        assert response.status_code == 200

        # State API is allowed
        state_response = await client.get("/state")
        assert state_response.status_code == 200
