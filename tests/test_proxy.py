from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from computecop.app import create_app
from computecop.config import EndpointConfig, RuntimeConfig
from computecop.models import EndpointKind, EndpointRoute, TelemetrySample, ThermalState, utc_now
from computecop.upstream import HealthProbe, UpstreamResponse


class FakeUpstream:
    def __init__(self) -> None:
        self.routes = {
            "ollama": EndpointRoute(
                name="ollama",
                kind=EndpointKind.OLLAMA,
                base_url="http://ollama.test",
                timeout_seconds=30.0,
                health_path="/api/tags",
            )
        }
        self.last_json: dict[str, Any] | None = None

    def route(self, name: str | None = None) -> EndpointRoute:
        return self.routes[name or "ollama"]

    async def probe(self, route: EndpointRoute | None = None) -> HealthProbe:
        return HealthProbe(endpoint=(route or self.route()).name, healthy=True, status_code=200, detail="OK")

    async def request(self, route, *, method, path, headers, json_body, content=None):
        self.last_json = json_body
        payload = {"path": path, "json": json_body}
        return UpstreamResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            content=json.dumps(payload).encode("utf-8"),
        )

    async def stream(self, route, *, method, path, headers, json_body):
        yield b"data: ok\n\n"

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_health_and_state_routes(tmp_path: Path) -> None:
    app = _app(tmp_path)
    fake = FakeUpstream()
    app.state.runtime.upstream = fake
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        health = await client.get("/health")
        state = await client.get("/state")
    assert health.status_code == 200
    assert health.json()["upstream"]["healthy"] is True
    assert state.status_code == 200
    assert state.json()["global_juice_level"] == 100


@pytest.mark.asyncio
async def test_openai_chat_shapes_budget(tmp_path: Path) -> None:
    app = _app(tmp_path)
    fake = FakeUpstream()
    app.state.runtime.upstream = fake
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "local", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 9999},
        )
    assert response.status_code == 200
    assert response.headers["x-computecop-decision"] == "allow"
    assert fake.last_json is not None
    assert fake.last_json["max_tokens"] <= 2048
    assert fake.last_json["metadata"]["computecop_juice_level"] == 100


@pytest.mark.asyncio
async def test_ollama_chat_shapes_options(tmp_path: Path) -> None:
    app = _app(tmp_path)
    fake = FakeUpstream()
    app.state.runtime.upstream = fake
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            headers={"x-computecop-background": "true"},
            json={"model": "mistral", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert fake.last_json is not None
    assert "num_ctx" in fake.last_json["options"]


@pytest.mark.asyncio
async def test_background_request_yields_under_ram_pressure(tmp_path: Path) -> None:
    app = _app(tmp_path)
    await app.state.runtime.state.update_telemetry(_telemetry(90.0))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/chat",
            headers={"x-computecop-background": "true"},
            json={"model": "mistral", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "computecop_yield"


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


def _telemetry(ram: float) -> TelemetrySample:
    return TelemetrySample(
        timestamp=utc_now(),
        cpu_percent=10.0,
        cpu_per_core_percent=(10.0,),
        ram_total_bytes=16 * 1024**3,
        ram_available_bytes=1 * 1024**3,
        ram_used_percent=ram,
        swap_used_percent=0.0,
        disk_read_bytes_per_sec=0.0,
        disk_write_bytes_per_sec=0.0,
        thermal_state=ThermalState.COOL,
    )
