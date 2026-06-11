from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from computecop.models import (
    EndpointKind,
    EndpointRoute,
    ProcessSample,
    RequestClass,
    TelemetrySample,
    ThermalState,
)
from computecop.offload import OffloadManager, rank_eviction_candidates
from computecop.residency import ModelResidency, ModelResidencyTracker, ResidencyStatus


def test_tracker_basic_request() -> None:
    tracker = ModelResidencyTracker(hot_threshold_seconds=1.0, warm_threshold_seconds=2.0)
    now = datetime.now(tz=UTC)

    # Initially empty
    assert tracker.get_estimates() == []

    # Record a request
    tracker.record_request("llama3", "endpoint1", RequestClass.USER_PROMPT, timestamp=now)
    estimates = tracker.get_estimates()
    assert len(estimates) == 1
    est = estimates[0]
    assert est.model == "llama3"
    assert est.endpoint == "endpoint1"
    assert est.status == ResidencyStatus.HOT
    assert est.confidence == 0.95
    assert est.last_request_class == RequestClass.USER_PROMPT


def test_tracker_telemetry_association() -> None:
    tracker = ModelResidencyTracker(hot_threshold_seconds=10.0, warm_threshold_seconds=60.0)
    now = datetime.now(tz=UTC)

    # Record request to set up active model association
    tracker.record_request("llama3", "ollama1", RequestClass.BACKGROUND_REQUEST, timestamp=now)

    # 1. First telemetry sample: sets baseline RSS for PID 101
    sample1 = TelemetrySample(
        timestamp=now,
        cpu_percent=10.0,
        cpu_per_core_percent=(10.0,),
        ram_total_bytes=16 * 1024 * 1024 * 1024,
        ram_available_bytes=8 * 1024 * 1024 * 1024,
        ram_used_percent=50.0,
        swap_used_percent=0.0,
        disk_read_bytes_per_sec=0.0,
        disk_write_bytes_per_sec=0.0,
        thermal_state=ThermalState.COOL,
        heavy_processes=(
            ProcessSample(
                pid=101, name="ollama runner", cpu_percent=50.0, memory_rss_bytes=100 * 1024 * 1024
            ),
        ),
    )
    tracker.update_from_telemetry(sample1)

    # 2. Second telemetry sample: RSS increases by 50MB (which is > 10MB)
    sample2 = TelemetrySample(
        timestamp=now,
        cpu_percent=10.0,
        cpu_per_core_percent=(10.0,),
        ram_total_bytes=16 * 1024 * 1024 * 1024,
        ram_available_bytes=8 * 1024 * 1024 * 1024,
        ram_used_percent=50.0,
        swap_used_percent=0.0,
        disk_read_bytes_per_sec=0.0,
        disk_write_bytes_per_sec=0.0,
        thermal_state=ThermalState.COOL,
        heavy_processes=(
            ProcessSample(
                pid=101, name="ollama runner", cpu_percent=50.0, memory_rss_bytes=150 * 1024 * 1024
            ),
        ),
    )
    tracker.update_from_telemetry(sample2)

    # Verify association
    estimates = tracker.get_estimates()
    assert len(estimates) == 1
    est = estimates[0]
    assert est.model == "llama3"
    assert est.estimated_memory_bytes == 150 * 1024 * 1024


@pytest.mark.anyio
async def test_tracker_active_polling() -> None:
    tracker = ModelResidencyTracker()
    routes = [
        EndpointRoute("ollama1", EndpointKind.OLLAMA, "http://ollama", 10.0, "/api/tags"),
        EndpointRoute("llamacpp1", EndpointKind.LLAMA_CPP, "http://llamacpp", 10.0, "/slots"),
    ]

    # Create mock response functions
    def mock_get(url: str, **kwargs: Any) -> httpx.Response:
        if "api/tags" in url:
            return httpx.Response(
                200,
                json={"models": [{"name": "llama3:latest", "size": 4000000000}]},
                request=httpx.Request("GET", url),
            )
        elif "api/ps" in url:
            return httpx.Response(
                200,
                json={"models": [{"name": "llama3:latest", "size": 4000000000}]},
                request=httpx.Request("GET", url),
            )
        elif "slots" in url:
            return httpx.Response(
                200,
                json=[{"id": 0, "model": "phi3.gguf"}],
                request=httpx.Request("GET", url),
            )
        return httpx.Response(404, request=httpx.Request("GET", url))

    # Mock client
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.get.side_effect = mock_get

    await tracker.update_from_endpoints(routes, mock_client)

    estimates = tracker.get_estimates()
    assert len(estimates) == 2

    # Parse by model
    phi3_est = next(e for e in estimates if "phi3" in e.model)
    llama3_est = next(e for e in estimates if "llama3" in e.model)

    assert phi3_est.endpoint == "llamacpp1"
    assert phi3_est.confidence == 1.0
    assert phi3_est.status == ResidencyStatus.HOT

    assert llama3_est.endpoint == "ollama1"
    assert llama3_est.confidence == 1.0  # From api/ps
    assert llama3_est.estimated_memory_bytes == 4000000000


def test_offload_candidate_ranking() -> None:
    now = datetime.now(tz=UTC)

    # 1. Active foreground model (phi3) - should be protected (last)
    phi3 = ModelResidency(
        model="phi3",
        endpoint="ep1",
        status=ResidencyStatus.HOT,
        confidence=1.0,
        last_accessed_at=now,
        last_request_class=RequestClass.USER_PROMPT,
        estimated_memory_bytes=2000000,
    )

    # 2. Background request, evictable, large size (llama3) - should be first candidate
    llama3 = ModelResidency(
        model="llama3",
        endpoint="ep1",
        status=ResidencyStatus.EVICTABLE,
        confidence=0.8,
        last_accessed_at=now,
        last_request_class=RequestClass.BACKGROUND_REQUEST,
        estimated_memory_bytes=4000000000,
    )

    # 3. Background request, warm, smaller size (qwen) - should be second candidate
    qwen = ModelResidency(
        model="qwen",
        endpoint="ep1",
        status=ResidencyStatus.WARM,
        confidence=0.9,
        last_accessed_at=now,
        last_request_class=RequestClass.BACKGROUND_REQUEST,
        estimated_memory_bytes=2000000000,
    )

    # 4. Cold model (gemma) - should be ranked after warm/evictable models, but before protected
    gemma = ModelResidency(
        model="gemma",
        endpoint="ep1",
        status=ResidencyStatus.COLD,
        confidence=0.1,
        last_accessed_at=None,
        last_request_class=None,
    )

    estimates = [phi3, llama3, qwen, gemma]
    ranked = rank_eviction_candidates(estimates, hot_threshold_seconds=60.0)

    # Expected order: llama3 (EVICTABLE, large), qwen (WARM, smaller),
    # gemma (COLD), phi3 (protected)
    assert ranked[0].model == "llama3"
    assert ranked[1].model == "qwen"
    assert ranked[2].model == "gemma"
    assert ranked[3].model == "phi3"


def test_offload_manager_rank_candidates() -> None:
    routes = [EndpointRoute("ep1", EndpointKind.OLLAMA, "http://ollama", 10.0, "/api/tags")]
    manager = OffloadManager(routes)

    estimates = [
        ModelResidency(
            model="m1",
            endpoint="ep1",
            status=ResidencyStatus.WARM,
            confidence=0.8,
            last_request_class=RequestClass.BACKGROUND_REQUEST,
        )
    ]
    ranked = manager.rank_candidates(estimates)
    assert len(ranked) == 1
    assert ranked[0].model == "m1"
