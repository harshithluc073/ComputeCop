from __future__ import annotations

import asyncio
from pathlib import Path
from time import monotonic
import httpx
import pytest

from computecop.app import create_app
from computecop.config import EndpointConfig, RuntimeConfig
from computecop.models import EndpointKind, EndpointRoute
from computecop.state import RuntimeStateStore
from tests.test_proxy import FakeUpstream, _app

@pytest.mark.asyncio
async def test_runtime_state_store_metrics() -> None:
    store = RuntimeStateStore()

    # Verify initial metrics are empty
    snapshot = await store.snapshot()
    assert "request_latency_histogram" in snapshot.metrics
    assert "queue_wait_time_histogram" in snapshot.metrics
    assert "upstream_duration_histogram" in snapshot.metrics
    assert "shaping" in snapshot.metrics

    # Test record request latency
    await store.record_request_latency(0.05) # bucket 0.1s
    await store.record_request_latency(0.4)  # bucket 0.5s
    await store.record_request_latency(12.0) # bucket 30.0s
    await store.record_request_latency(70.0) # bucket > 60s (+inf)

    # Test record queue wait time
    await store.record_queue_wait_time(0.08)
    await store.record_queue_wait_time(2.0)

    # Test record upstream duration
    await store.record_upstream_duration(0.3)
    await store.record_upstream_duration(55.0)

    # Test record shaping
    await store.record_shaping(shaped=True)
    await store.record_shaping(shaped=False)
    await store.record_shaping(shaped=True)

    snapshot = await store.snapshot()
    metrics = snapshot.metrics

    # Verify request latency histogram counts
    # Buckets: 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0 (8 boundaries -> 9 bins)
    latency_counts = metrics["request_latency_histogram"]["counts"]
    assert latency_counts[0] == 1  # <= 0.1 (0.05s)
    assert latency_counts[1] == 1  # <= 0.5 (0.4s)
    assert latency_counts[6] == 1  # <= 30.0 (12.0s)
    assert latency_counts[8] == 1  # > 60.0 (70.0s)

    # Verify queue wait time counts
    wait_counts = metrics["queue_wait_time_histogram"]["counts"]
    assert wait_counts[0] == 1  # <= 0.1 (0.08s)
    assert wait_counts[3] == 1  # <= 2.5 (2.0s)

    # Verify shaping counters
    shaping = metrics["shaping"]
    assert shaping["total_requests"] == 3
    assert shaping["shaped_requests"] == 2
    assert shaping["shaping_ratio"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_metrics_api_endpoint(tmp_path: Path) -> None:
    app = _app(tmp_path)
    fake = FakeUpstream()
    app.state.runtime.upstream = fake

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Make a request that is not shaped
        resp1 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10,
            },
        )
        assert resp1.status_code == 200

        # Make a request that gets shaped
        resp2 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 9999,
            },
        )
        assert resp2.status_code == 200

        # Retrieve /metrics
        metrics_resp = await client.get("/metrics")
        assert metrics_resp.status_code == 200
        metrics = metrics_resp.json()

        assert "request_latency_histogram" in metrics
        assert "queue_wait_time_histogram" in metrics
        assert "upstream_duration_histogram" in metrics
        assert "shaping" in metrics

        shaping = metrics["shaping"]
        assert shaping["total_requests"] == 2
        assert shaping["shaped_requests"] == 1
        assert shaping["shaping_ratio"] == 0.5


@pytest.mark.asyncio
async def test_streaming_does_not_buffer(tmp_path: Path) -> None:
    app = _app(tmp_path)
    fake = FakeUpstream()
    app.state.runtime.upstream = fake

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Request with streaming enabled
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            chunks = [chunk async for chunk in response.aiter_bytes()]
            assert len(chunks) > 0
            assert b"data: ok" in chunks[0]

        # Retrieve /metrics to verify upstream duration and shaping were recorded
        metrics_resp = await client.get("/metrics")
        assert metrics_resp.status_code == 200
        metrics = metrics_resp.json()

        # Upstream duration counts should have recorded 1 event
        upstream_counts = metrics["upstream_duration_histogram"]["counts"]
        assert sum(upstream_counts) == 1
