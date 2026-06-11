"""Best-effort model offload adapters."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from computecop.logging import get_logger, log_event
from computecop.models import EndpointKind, EndpointRoute, RequestClass, utc_now
from computecop.residency import ModelResidency, ResidencyStatus


@dataclass(frozen=True, slots=True)
class OffloadResult:
    """Result from an offload attempt."""

    endpoint: str
    attempted: bool
    succeeded: bool
    detail: str


class OffloadManager:
    """Coordinate offload attempts across configured endpoints."""

    def __init__(self, routes: list[EndpointRoute]) -> None:
        self.routes = routes
        self._logger = get_logger("offload")

    async def offload_all(self, reason: str) -> tuple[OffloadResult, ...]:
        """Attempt to free model memory on every supported endpoint."""

        results: list[OffloadResult] = []
        async with httpx.AsyncClient(timeout=10.0) as client:
            for route in self.routes:
                adapter = adapter_for_route(route, client)
                result = await adapter.offload(reason)
                results.append(result)
                level = logging.INFO if result.succeeded else logging.WARNING
                log_event(
                    self._logger,
                    level,
                    "model offload attempted",
                    endpoint=result.endpoint,
                    succeeded=result.succeeded,
                    detail=result.detail,
                )
        return tuple(results)

    def rank_candidates(
        self,
        residency_estimates: list[ModelResidency],
        hot_threshold_seconds: float = 60.0,
    ) -> list[ModelResidency]:
        """Rank eviction candidates from best to worst/protected."""
        return rank_eviction_candidates(residency_estimates, hot_threshold_seconds)


class BaseOffloadAdapter:
    """Base adapter for an upstream endpoint family."""

    def __init__(self, route: EndpointRoute, client: httpx.AsyncClient) -> None:
        self.route = route
        self.client = client

    async def offload(self, reason: str) -> OffloadResult:
        return OffloadResult(
            endpoint=self.route.name,
            attempted=False,
            succeeded=False,
            detail=f"offload unsupported for {self.route.kind.value}: {reason}",
        )


class OllamaOffloadAdapter(BaseOffloadAdapter):
    """Unload Ollama models with keep_alive=0 best-effort calls."""

    async def offload(self, reason: str) -> OffloadResult:
        try:
            tags = await self.client.get(f"{self.route.base_url}/api/tags")
            tags.raise_for_status()
            models = _ollama_model_names(tags.json())
            if not models:
                return OffloadResult(self.route.name, True, True, "no Ollama models reported")

            failures: list[str] = []
            for model in models:
                response = await self.client.post(
                    f"{self.route.base_url}/api/generate",
                    json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
                )
                if response.status_code >= 400:
                    failures.append(f"{model}:{response.status_code}")
            if failures:
                return OffloadResult(
                    self.route.name,
                    True,
                    False,
                    f"Ollama unload failed for {', '.join(failures)}",
                )
            return OffloadResult(
                self.route.name,
                True,
                True,
                f"requested Ollama unload for {len(models)} model(s)",
            )
        except (httpx.HTTPError, ValueError) as exc:
            return OffloadResult(self.route.name, True, False, f"Ollama offload failed: {exc!r}")


class LlamaCppOffloadAdapter(BaseOffloadAdapter):
    """Best-effort llama.cpp context clearing via slots API."""

    async def offload(self, reason: str) -> OffloadResult:
        try:
            response = await self.client.get(f"{self.route.base_url}/slots")
            if response.status_code == 404:
                return OffloadResult(
                    self.route.name,
                    True,
                    False,
                    "llama.cpp slots API unavailable",
                )
            response.raise_for_status()
            slots = response.json()
            if not isinstance(slots, list) or not slots:
                return OffloadResult(self.route.name, True, True, "no llama.cpp slots reported")

            failures: list[str] = []
            for slot in slots:
                slot_id = _slot_id(slot)
                if slot_id is None:
                    continue
                clear = await self.client.post(
                    f"{self.route.base_url}/slots/{slot_id}",
                    params={"action": "erase"},
                )
                if clear.status_code >= 400:
                    failures.append(f"slot-{slot_id}:{clear.status_code}")
            if failures:
                return OffloadResult(
                    self.route.name,
                    True,
                    False,
                    f"llama.cpp slot clear failed for {', '.join(failures)}",
                )
            return OffloadResult(
                self.route.name, True, True, f"cleared {len(slots)} llama.cpp slot(s)"
            )
        except (httpx.HTTPError, ValueError) as exc:
            return OffloadResult(self.route.name, True, False, f"llama.cpp offload failed: {exc!r}")


def adapter_for_route(route: EndpointRoute, client: httpx.AsyncClient) -> BaseOffloadAdapter:
    if route.kind == EndpointKind.OLLAMA:
        return OllamaOffloadAdapter(route, client)
    if route.kind == EndpointKind.LLAMA_CPP:
        return LlamaCppOffloadAdapter(route, client)
    return BaseOffloadAdapter(route, client)


def _ollama_model_names(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    models = payload.get("models")
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for item in models:
        if isinstance(item, dict) and item.get("name"):
            names.append(str(item["name"]))
    return names


def _slot_id(slot: object) -> int | None:
    if not isinstance(slot, dict):
        return None
    for key in ("id", "id_slot"):
        if key in slot:
            try:
                return int(slot[key])
            except (TypeError, ValueError):
                return None
    return None


def rank_eviction_candidates(
    residency_estimates: list[ModelResidency],
    hot_threshold_seconds: float = 60.0,
) -> list[ModelResidency]:
    """Rank models from best eviction candidate (first) to worst/protected candidate (last)."""

    now = utc_now()

    def get_sort_key(est: ModelResidency) -> tuple[bool, int, int, datetime, int, float]:
        # 1. Protected status (active foreground model)
        is_protected = False
        if est.last_request_class == RequestClass.USER_PROMPT and est.last_accessed_at is not None:
            age = (now - est.last_accessed_at).total_seconds()
            if age < hot_threshold_seconds:
                is_protected = True

        # 2. Status priority (EVICTABLE = 0, WARM = 1, HOT = 2, COLD = 3)
        status_priority = 3
        if est.status == ResidencyStatus.EVICTABLE:
            status_priority = 0
        elif est.status == ResidencyStatus.WARM:
            status_priority = 1
        elif est.status == ResidencyStatus.HOT:
            status_priority = 2
        elif est.status == ResidencyStatus.COLD:
            status_priority = 3

        # 3. Request class priority (BACKGROUND = 0, UNKNOWN/None = 1, USER_PROMPT = 2)
        class_priority = 1
        if est.last_request_class == RequestClass.BACKGROUND_REQUEST:
            class_priority = 0
        elif est.last_request_class == RequestClass.USER_PROMPT:
            class_priority = 2

        # 4. Access recency (older/None first)
        access_time = est.last_accessed_at or datetime.min.replace(tzinfo=now.tzinfo)

        # 5. Estimated memory size (larger size first -> negative of size)
        size = est.estimated_memory_bytes or 0
        neg_size = -size

        # 6. Confidence (higher confidence first -> negative of confidence)
        neg_conf = -est.confidence

        return (is_protected, status_priority, class_priority, access_time, neg_size, neg_conf)

    return sorted(residency_estimates, key=get_sort_key)
