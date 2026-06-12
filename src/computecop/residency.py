"""Model residency tracking and status estimation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import httpx

from computecop.logging import get_logger
from computecop.models import EndpointKind, RequestClass, utc_now

if TYPE_CHECKING:
    from computecop.models import EndpointRoute, TelemetrySample


class ResidencyStatus(str, Enum):
    """Estimated state of a model in memory."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    EVICTABLE = "evictable"


@dataclass(frozen=True, slots=True)
class ModelResidency:
    """Estimated residency state for a model on an endpoint."""

    model: str
    endpoint: str
    status: ResidencyStatus
    confidence: float  # 0.0 to 1.0
    last_accessed_at: datetime | None = None
    estimated_memory_bytes: int | None = None
    inferred_from: str = "default"
    last_request_class: RequestClass | None = None


class ModelResidencyTracker:
    """Tracks active and inactive models across upstream endpoints."""

    def __init__(
        self,
        hot_threshold_seconds: float = 60.0,
        warm_threshold_seconds: float = 300.0,
    ) -> None:
        self.hot_threshold_seconds = hot_threshold_seconds
        self.warm_threshold_seconds = warm_threshold_seconds
        self._logger = get_logger("residency")

        # In-memory tracking
        self._last_accessed: dict[tuple[str, str], datetime] = {}
        self._last_request_class: dict[tuple[str, str], RequestClass] = {}
        self._model_endpoints: dict[str, set[str]] = {}  # model -> endpoints

        # Endpoint observations from active polling
        # (model, endpoint) -> (source, size_bytes, confidence, timestamp)
        self._endpoint_observations: dict[
            tuple[str, str], tuple[str, int | None, float, datetime]
        ] = {}

        # Process RSS tracking
        self._last_process_rss: dict[int, int] = {}
        self._process_model_association: dict[int, str] = {}  # pid -> model
        self._estimated_sizes: dict[tuple[str, str], int] = {}

    def record_request(
        self,
        model: str,
        endpoint: str,
        request_class: RequestClass,
        timestamp: datetime | None = None,
    ) -> None:
        """Record an incoming request to update access recency."""

        ts = timestamp or utc_now()
        key = (model, endpoint)
        self._last_accessed[key] = ts
        self._last_request_class[key] = request_class
        if model not in self._model_endpoints:
            self._model_endpoints[model] = set()
        self._model_endpoints[model].add(endpoint)

    def update_from_telemetry(self, telemetry: TelemetrySample) -> None:
        """Infer model residency from process RSS deltas and heavy processes."""

        current_pids = set()
        now = utc_now()

        for proc in telemetry.heavy_processes:
            pid = proc.pid
            name = proc.name.lower()
            current_pids.add(pid)

            # Identify if this is a model runner process
            is_runner = "ollama" in name or "llama" in name
            if not is_runner:
                continue

            prev_rss = self._last_process_rss.get(pid, 0)
            curr_rss = proc.memory_rss_bytes
            self._last_process_rss[pid] = curr_rss

            # Compute delta
            delta = curr_rss - prev_rss

            # If RSS increased significantly, check if there was a recent request
            if delta > 10 * 1024 * 1024:  # > 10MB delta
                best_model = None
                best_endpoint = None
                best_time = None

                for (m, e), last_ts in self._last_accessed.items():
                    if (now - last_ts).total_seconds() < 30:
                        if best_time is None or last_ts > best_time:
                            best_time = last_ts
                            best_model = m
                            best_endpoint = e

                if best_model and best_endpoint:
                    self._process_model_association[pid] = best_model
                    self._estimated_sizes[(best_model, best_endpoint)] = curr_rss
                    self._logger.info(
                        "Associated process %d (%s) RSS delta %d with model %s on %s",
                        pid,
                        proc.name,
                        delta,
                        best_model,
                        best_endpoint,
                    )

            # If we already have an associated model, update its estimated size
            associated_model = self._process_model_association.get(pid)
            if associated_model:
                endpoints = self._model_endpoints.get(associated_model, set())
                for e in endpoints:
                    self._estimated_sizes[(associated_model, e)] = curr_rss

        # Clean up dead PIDs
        dead_pids = set(self._last_process_rss.keys()) - current_pids
        for pid in dead_pids:
            self._last_process_rss.pop(pid, None)
            self._process_model_association.pop(pid, None)

    async def update_from_endpoints(
        self,
        routes: list[EndpointRoute],
        client: httpx.AsyncClient,
    ) -> None:
        """Actively query endpoints to discover loaded or available models."""

        now = utc_now()
        for route in routes:
            if route.kind == EndpointKind.OLLAMA:
                # Query tags (available models)
                try:
                    tags_resp = await client.get(
                        f"{route.base_url}/api/tags",
                        timeout=5.0,
                    )
                    if tags_resp.status_code == 200:
                        tags_data = tags_resp.json()
                        models = tags_data.get("models", [])
                        for item in models:
                            name = item.get("name")
                            if name:
                                size = item.get("size")
                                key = (name, route.name)
                                self._endpoint_observations[key] = ("ollama_tags", size, 0.2, now)
                except Exception as exc:
                    self._logger.debug("Failed to query Ollama tags: %r", exc)

                # Query ps (currently loaded models in memory)
                try:
                    ps_resp = await client.get(
                        f"{route.base_url}/api/ps",
                        timeout=5.0,
                    )
                    if ps_resp.status_code == 200:
                        ps_data = ps_resp.json()
                        models = ps_data.get("models", [])
                        for item in models:
                            name = item.get("name")
                            if name:
                                size = item.get("size")
                                key = (name, route.name)
                                self._endpoint_observations[key] = ("ollama_ps", size, 1.0, now)
                except Exception as exc:
                    self._logger.debug("Failed to query Ollama ps: %r", exc)

            elif route.kind == EndpointKind.LLAMA_CPP:
                # Query slots (tells us what is loaded)
                try:
                    slots_resp = await client.get(
                        f"{route.base_url}/slots",
                        timeout=5.0,
                    )
                    if slots_resp.status_code == 200:
                        slots = slots_resp.json()
                        if isinstance(slots, list):
                            for slot in slots:
                                model_path = slot.get("model")
                                if model_path:
                                    key = (model_path, route.name)
                                    self._endpoint_observations[key] = (
                                        "llamacpp_slots",
                                        None,
                                        1.0,
                                        now,
                                    )
                except Exception as exc:
                    self._logger.debug("Failed to query llama.cpp slots: %r", exc)

            elif route.kind == EndpointKind.OPENAI_COMPATIBLE:
                # Query /v1/models (available models)
                try:
                    models_resp = await client.get(
                        f"{route.base_url}/v1/models",
                        timeout=5.0,
                    )
                    if models_resp.status_code == 200:
                        models_data = models_resp.json()
                        models = models_data.get("data", [])
                        for item in models:
                            name = item.get("id")
                            if name:
                                key = (name, route.name)
                                self._endpoint_observations[key] = ("openai_models", None, 0.2, now)
                except Exception as exc:
                    self._logger.debug("Failed to query OpenAI compatible models: %r", exc)

    def get_estimates(self) -> list[ModelResidency]:
        """Aggregate observations and access history to estimate residency."""

        estimates: dict[tuple[str, str], ModelResidency] = {}
        now = utc_now()

        all_keys = set(self._last_accessed.keys()) | set(self._endpoint_observations.keys())

        for model, endpoint in all_keys:
            key = (model, endpoint)
            last_ts = self._last_accessed.get(key)
            req_class = self._last_request_class.get(key)
            obs = self._endpoint_observations.get(key)

            confidence = 0.0
            inferred_from = "default"
            est_size = self._estimated_sizes.get(key)

            if obs:
                obs_src, obs_size, obs_conf, obs_ts = obs
                age = (now - obs_ts).total_seconds()
                if age > 300:
                    obs_conf *= max(0.0, 1.0 - (age - 300) / 300)

                if obs_conf > confidence:
                    confidence = obs_conf
                    inferred_from = obs_src
                if obs_size is not None:
                    est_size = obs_size

            if last_ts:
                seconds_since_request = (now - last_ts).total_seconds()
                req_conf = 0.0
                if seconds_since_request < self.hot_threshold_seconds:
                    req_conf = 0.95
                elif seconds_since_request < self.warm_threshold_seconds:
                    req_conf = 0.70
                elif seconds_since_request < 1800:
                    req_conf = 0.30

                if req_conf > confidence:
                    confidence = req_conf
                    inferred_from = "recent_request"

            status = ResidencyStatus.COLD

            if last_ts and (now - last_ts).total_seconds() < self.hot_threshold_seconds:
                status = ResidencyStatus.HOT
            elif confidence >= 0.9:
                status = ResidencyStatus.HOT
            elif last_ts and (now - last_ts).total_seconds() < self.warm_threshold_seconds:
                status = ResidencyStatus.WARM
            elif confidence >= 0.5:
                status = ResidencyStatus.WARM

            if status == ResidencyStatus.WARM:
                if req_class == RequestClass.BACKGROUND_REQUEST:
                    status = ResidencyStatus.EVICTABLE
                elif last_ts and (now - last_ts).total_seconds() >= self.hot_threshold_seconds:
                    status = ResidencyStatus.EVICTABLE
                elif not last_ts:
                    status = ResidencyStatus.EVICTABLE

            estimates[key] = ModelResidency(
                model=model,
                endpoint=endpoint,
                status=status,
                confidence=confidence,
                last_accessed_at=last_ts,
                estimated_memory_bytes=est_size,
                inferred_from=inferred_from,
                last_request_class=req_class,
            )

        return list(estimates.values())

    def is_model_compatible(self, model: str, endpoint: str) -> bool:
        """Return whether the endpoint is compatible with the requested model."""
        observed_models = {m for m, ep in self._endpoint_observations if ep == endpoint}
        accessed_models = {m for m, ep in self._last_accessed if ep == endpoint}
        all_models = observed_models | accessed_models

        if not all_models:
            return True  # No observations, treat as compatible fallback

        for obs in all_models:
            if model_matches(model, obs):
                return True
        return False


def model_matches(requested: str, observed: str) -> bool:
    """Check if requested model matches observed model, ignoring tags, paths, and extensions."""
    req = requested.strip().lower()
    obs = observed.strip().lower()
    if req == obs:
        return True

    # Strip tags (e.g., llama3:8b -> llama3)
    req_base = req.split(":")[0]
    obs_base = obs.split(":")[0]
    if req_base == obs_base:
        return True

    # Strip paths and extensions (e.g., /models/llama-3.gguf -> llama-3)
    import os

    req_name, _ = os.path.splitext(os.path.basename(req_base))
    obs_name, _ = os.path.splitext(os.path.basename(obs_base))
    if req_name == obs_name:
        return True

    # Helper to normalize delimiters like - and _
    def normalize(s: str) -> str:
        return s.replace("-", "").replace("_", "").replace(".", "")

    req_norm = normalize(req_name)
    obs_norm = normalize(obs_name)
    if req_norm == obs_norm:
        return True
    if req_norm in obs_norm or obs_norm in req_norm:
        return True

    return False
