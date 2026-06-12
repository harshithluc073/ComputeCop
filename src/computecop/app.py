"""FastAPI application factory for the ComputeCop proxy."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import monotonic
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from computecop import __version__
from computecop.admission import AdmissionController
from computecop.classifier import RequestClassifier
from computecop.concurrency import EndpointConcurrencyGovernor, is_foreground_metadata
from computecop.config import RuntimeConfig, cached_config
from computecop.endpoints import EndpointCapabilityRegistry, resolve_api_family
from computecop.events import JsonlEventStore
from computecop.health import EndpointHealthWatcher
from computecop.models import DecisionType, RequestClass, RequestMetadata, to_jsonable
from computecop.offload import OffloadManager
from computecop.platform import current_platform_name
from computecop.policy import JuicePolicyEngine
from computecop.processes import HeavyProcessDetector
from computecop.request_queue import AsyncRequestQueue, QueueFullError, QueueTimeoutError
from computecop.responses import decision_headers, decision_response, error_response
from computecop.scheduler import AdaptiveScheduler
from computecop.state import RuntimeStateStore
from computecop.telemetry import PsutilTelemetrySampler, total_ram_bytes
from computecop.telemetry_loop import TelemetryLoop
from computecop.thermal import ThermalDetector, ThermalThresholds
from computecop.upstream import UpstreamFailure, UpstreamFailureCategory, UpstreamRouter
from computecop.yielding import RamYieldController


@dataclass(slots=True)
class ComputeCopRuntime:
    """Application runtime dependencies."""

    config: RuntimeConfig
    state: RuntimeStateStore
    classifier: RequestClassifier
    policy: JuicePolicyEngine
    admission: AdmissionController
    queue: AsyncRequestQueue
    scheduler: AdaptiveScheduler
    concurrency_governor: EndpointConcurrencyGovernor
    upstream: UpstreamRouter
    endpoint_registry: EndpointCapabilityRegistry
    health_watcher: EndpointHealthWatcher
    telemetry_loop: TelemetryLoop
    yield_controller: RamYieldController
    offload_manager: OffloadManager
    event_store: JsonlEventStore
    queue_workers: list[asyncio.Task[None]]
    _stopping: bool = field(default=False, init=False)
    _stopped: bool = field(default=False, init=False)

    async def start(self) -> None:
        """Start background runtime services."""

        await self.telemetry_loop.start()
        await self.health_watcher.start()
        if not self.queue_workers:
            await self.scheduler.start()
            self.queue_workers = list(self.scheduler.worker_tasks)

    async def stop(self, *, drain_timeout_seconds: float | None = None) -> None:
        """Stop background runtime services."""

        if self._stopping or self._stopped:
            return
        self._stopping = True
        try:
            await self.health_watcher.stop()
            await self.telemetry_loop.stop()
            drain_seconds = (
                self.config.queue.shutdown_drain_seconds
                if drain_timeout_seconds is None
                else drain_timeout_seconds
            )
            if drain_seconds > 0:
                await self.queue.drain(monotonic() + drain_seconds)
            await self.queue.close()
            await self.scheduler.stop()
            self.queue_workers.clear()
            await self.upstream.close()
        finally:
            self._stopped = True
            self._stopping = False


def build_runtime(config: RuntimeConfig) -> ComputeCopRuntime:
    """Build the runtime dependency graph."""

    thresholds = ThermalThresholds(
        warm_celsius=config.policy.thermal_warm_celsius,
        hot_celsius=config.policy.thermal_hot_celsius,
        critical_celsius=config.policy.thermal_critical_celsius,
    )
    sampler = PsutilTelemetrySampler(
        thermal_detector=ThermalDetector(thresholds),
        process_detector=HeavyProcessDetector.for_host(
            total_ram_bytes=total_ram_bytes(),
            limit=config.telemetry.heavy_process_limit,
        ),
    )
    state = RuntimeStateStore()
    policy = JuicePolicyEngine(config.policy)
    routes = [endpoint.to_route() for endpoint in config.endpoints]
    upstream = UpstreamRouter(routes)
    registry_config = config.endpoint_registry
    endpoint_registry = EndpointCapabilityRegistry(
        upstream,
        probe_ttl_seconds=registry_config.capability_probe_ttl_seconds,
        failure_threshold=registry_config.circuit_breaker_failure_threshold,
        cooldown_seconds=registry_config.circuit_breaker_cooldown_seconds,
        half_open_successes=registry_config.circuit_breaker_half_open_successes,
        residency_tracker=state.residency_tracker,
    )
    health_watcher = EndpointHealthWatcher(
        endpoint_registry,
        upstream,
        interval_seconds=registry_config.health_watcher_interval_seconds,
        jitter_fraction=registry_config.health_watcher_jitter_fraction,
        enabled=registry_config.health_watcher_enabled,
        residency_tracker=state.residency_tracker,
    )
    queue = AsyncRequestQueue(config.queue)
    queue.set_change_callback(state.update_queue)
    scheduler = AdaptiveScheduler(
        queue,
        policy_config=config.policy,
        queue_config=config.queue,
    )
    scheduler.set_change_callback(state.update_scheduler)
    concurrency_governor = EndpointConcurrencyGovernor(
        [endpoint.name for endpoint in config.endpoints]
    )
    concurrency_governor.set_change_callback(state.update_concurrency)
    yield_controller = RamYieldController(config.policy)
    offload_manager = OffloadManager(routes)
    event_store = JsonlEventStore(config.event_log_path)

    async def on_event_persistence_change(enabled: bool, reason: str | None) -> None:
        await state.set_event_persistence(enabled=enabled, disabled_reason=reason)

    event_store.set_persistence_callback(on_event_persistence_change)

    async def offload_hook(reason: str) -> None:
        await offload_manager.offload_all(reason)

    yield_controller.register_offload_hook(offload_hook)
    telemetry_loop = TelemetryLoop(
        sampler=sampler,
        interval_seconds=config.telemetry.interval_seconds,
        smoothing_window=config.telemetry.smoothing_window,
    )

    async def update_runtime(sample):
        await state.update_telemetry(sample)
        state.residency_tracker.update_from_telemetry(sample)
        await yield_controller.update(sample)
        pressure = policy.evaluate(
            sample,
            open_circuit_breaker_count=endpoint_registry.open_circuit_breaker_count(),
        )
        await concurrency_governor.update_limits(pressure.concurrency_limits)
        await state.set_policy_state(
            system_state=pressure.system_state,
            global_juice_level=pressure.global_juice_level,
            yield_active=pressure.yield_active,
            yield_reason=pressure.yield_reason,
        )
        if pressure.yield_active:
            await event_store.append("policy.yield", reason=pressure.yield_reason)
        await scheduler.update_pressure(pressure)
        await state.update_scheduler(scheduler.snapshot())
        await state.update_concurrency(concurrency_governor.snapshot())

    telemetry_loop.subscribe(update_runtime)

    return ComputeCopRuntime(
        config=config,
        state=state,
        classifier=RequestClassifier(),
        policy=policy,
        admission=AdmissionController(policy, config.queue),
        queue=queue,
        scheduler=scheduler,
        concurrency_governor=concurrency_governor,
        upstream=upstream,
        endpoint_registry=endpoint_registry,
        health_watcher=health_watcher,
        telemetry_loop=telemetry_loop,
        yield_controller=yield_controller,
        offload_manager=offload_manager,
        event_store=event_store,
        queue_workers=[],
    )


def create_app(config: RuntimeConfig | None = None) -> FastAPI:
    """Create the ComputeCop FastAPI app."""

    runtime = build_runtime(config or cached_config())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = runtime
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    app = FastAPI(
        title="ComputeCop",
        version=__version__,
        description="Local inference traffic controller with telemetry-aware compute budgeting.",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.get("/health")
    async def health() -> dict[str, object]:
        probe = await runtime.upstream.probe()
        return {
            "ok": True,
            "platform": current_platform_name(),
            "upstream": to_jsonable(probe),
        }

    @app.get("/state")
    async def state() -> dict[str, object]:
        snapshot = await runtime.state.snapshot()
        return snapshot.to_dict()

    @app.get("/telemetry")
    async def telemetry() -> dict[str, object]:
        snapshot = await runtime.state.snapshot()
        return {"telemetry": to_jsonable(snapshot.telemetry)}

    @app.get("/events")
    async def events(limit: int = 100) -> dict[str, object]:
        return {"events": list(await runtime.event_store.tail(limit=max(1, min(limit, 500))))}

    @app.get("/endpoints")
    async def endpoints(refresh: bool = False) -> dict[str, object]:
        records = await runtime.endpoint_registry.list_records(force_probe=refresh)
        return {"endpoints": [record.to_dict() for record in records]}

    @app.get("/queue/inspect")
    async def inspect_queue() -> dict[str, object]:
        items = await runtime.queue.inspect()
        return {"queue": items}

    @app.post("/queue/pause")
    async def pause_queue() -> dict[str, object]:
        await runtime.queue.pause()
        return {"ok": True, "state": runtime.queue.lifecycle_state.value}

    @app.post("/queue/resume")
    async def resume_queue() -> dict[str, object]:
        await runtime.queue.resume()
        return {"ok": True, "state": runtime.queue.lifecycle_state.value}

    @app.get("/decisions/{correlation_id}")
    async def decision(correlation_id: str) -> Response:
        found = await runtime.state.decision_for_correlation_id(correlation_id)
        if found is None:
            return error_response(
                status_code=404,
                message=f"decision '{correlation_id}' was not found in recent history",
                error_type="computecop_decision_not_found",
                correlation_id=correlation_id,
            )
        return Response(
            content=json_dumps({"decision": to_jsonable(found)}),
            media_type="application/json",
            headers=decision_headers(found),
        )

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request) -> Response:
        return await _handle_inference_request(
            runtime=runtime,
            request=request,
            upstream_path="/v1/chat/completions",
            family="openai",
        )

    @app.post("/api/generate")
    async def ollama_generate(request: Request) -> Response:
        return await _handle_inference_request(
            runtime=runtime,
            request=request,
            upstream_path="/api/generate",
            family="ollama",
        )

    @app.post("/api/chat")
    async def ollama_chat(request: Request) -> Response:
        return await _handle_inference_request(
            runtime=runtime,
            request=request,
            upstream_path="/api/chat",
            family="ollama",
        )

    @app.api_route("/v1/{proxy_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    async def v1_passthrough(proxy_path: str, request: Request) -> Response:
        return await _handle_inference_request(
            runtime=runtime,
            request=request,
            upstream_path=f"/v1/{proxy_path}",
            family="openai",
        )

    @app.post("/completion")
    async def llama_completion(request: Request) -> Response:
        return await _handle_inference_request(
            runtime=runtime,
            request=request,
            upstream_path="/completion",
            family="llama_cpp",
        )

    @app.post("/chat/completions")
    async def llama_chat_completions(request: Request) -> Response:
        return await _handle_inference_request(
            runtime=runtime,
            request=request,
            upstream_path="/chat/completions",
            family="llama_cpp",
        )

    return app


app = create_app()


async def _handle_inference_request(
    *,
    runtime: ComputeCopRuntime,
    request: Request,
    upstream_path: str,
    family: str,
) -> Response:
    body = await _json_body(request)
    metadata = runtime.classifier.classify(
        method=request.method,
        path=str(request.url.path),
        headers=request.headers,
        body=body if isinstance(body, dict) else None,
        client_host=request.client.host if request.client else None,
    )
    snapshot = await runtime.state.snapshot()
    pressure = runtime.policy.evaluate(
        snapshot.telemetry,
        open_circuit_breaker_count=runtime.endpoint_registry.open_circuit_breaker_count(),
    )
    decision = runtime.admission.decide(
        metadata,
        pressure,
        queue_size=runtime.queue.snapshot().queued,
    )
    await runtime.state.record_decision(decision)
    await runtime.event_store.append(
        "admission.decision",
        decision=decision,
        trace_id=decision.trace.trace_id if decision.trace else None,
        path=metadata.path,
        model=metadata.model,
        endpoint=metadata.endpoint_name,
    )

    if decision.decision in {DecisionType.REJECT, DecisionType.YIELD}:
        return decision_response(decision, status_code=429 if decision.retry_after_seconds else 503)

    async def forward() -> Response:
        return await _forward_upstream(
            runtime=runtime,
            request=request,
            metadata=metadata,
            decision=decision,
            upstream_path=upstream_path,
            family=family,
            body=body,
        )

    if (
        decision.decision == DecisionType.THROTTLE
        and metadata.request_class == RequestClass.BACKGROUND_REQUEST
    ):
        if not runtime.queue.accepts_background_work():
            queue_state = runtime.queue.lifecycle_state.value
            return error_response(
                status_code=503,
                message=f"background queue is {queue_state} and not accepting work",
                error_type="computecop_queue_unavailable",
                correlation_id=metadata.correlation_id,
                retry_after_seconds=runtime.config.queue.background_retry_after_seconds,
            )
        try:
            return await runtime.scheduler.execute_queued(metadata, forward)
        except QueueFullError as exc:
            status_code = 429 if "full" in str(exc) else 503
            error_type = (
                "computecop_queue_full" if status_code == 429 else "computecop_queue_unavailable"
            )
            return error_response(
                status_code=status_code,
                message=str(exc),
                error_type=error_type,
                correlation_id=metadata.correlation_id,
                retry_after_seconds=runtime.config.queue.background_retry_after_seconds,
            )
        except QueueTimeoutError:
            return error_response(
                status_code=504,
                message="queued background request expired before execution",
                error_type="computecop_queue_timeout",
                correlation_id=metadata.correlation_id,
                retry_after_seconds=runtime.config.queue.background_retry_after_seconds,
            )

    return await runtime.scheduler.execute_immediate(metadata, forward)


async def _forward_upstream(
    *,
    runtime: ComputeCopRuntime,
    request: Request,
    metadata: RequestMetadata,
    decision,
    upstream_path: str,
    family: str,
    body: Any,
) -> Response:
    shaping_headers: dict[str, str] = {}
    if isinstance(body, dict):
        shaped_body, shaping_headers = _shape_body(family, body, decision.budget)
    else:
        shaped_body = body
    headers = dict(request.headers)
    headers["x-computecop-correlation-id"] = metadata.correlation_id
    headers["x-computecop-juice-level"] = str(decision.budget.juice_level)

    foreground = is_foreground_metadata(metadata)
    requires_streaming = isinstance(shaped_body, dict) and shaped_body.get("stream") is True

    def make_release_capacity(route_name: str, fg: bool) -> Callable[[], Awaitable[None]]:
        async def release_capacity() -> None:
            await runtime.concurrency_governor.release(route_name, foreground=fg)

        return release_capacity

    can_retry = not requires_streaming and not metadata.endpoint_name
    failed_endpoints: set[str] = set()

    while True:
        route = None
        capacity_acquired = False
        try:
            route = _select_route(
                runtime,
                metadata.endpoint_name,
                family,
                model=metadata.model,
                requires_streaming=requires_streaming,
                exclude=failed_endpoints,
            )
            if metadata.model:
                runtime.state.residency_tracker.record_request(
                    metadata.model,
                    route.name,
                    metadata.request_class,
                )
            await runtime.concurrency_governor.acquire(route.name, foreground=foreground)
            capacity_acquired = True

            if requires_streaming and route.supports_streaming:
                return StreamingResponse(
                    _tracked_stream(
                        runtime=runtime,
                        route_name=route.name,
                        stream=runtime.upstream.stream(
                            route,
                            method=request.method,
                            path=upstream_path,
                            headers=headers,
                            json_body=shaped_body,
                        ),
                        endpoint_release=make_release_capacity(route.name, foreground),
                    ),
                    media_type="text/event-stream",
                    headers={
                        **decision_headers(decision),
                        **shaping_headers,
                    },
                )

            # Non-streaming request
            upstream = await runtime.upstream.request(
                route,
                method=request.method,
                path=upstream_path,
                headers=headers,
                json_body=shaped_body,
            )
            runtime.endpoint_registry.record_upstream_success(route.name)
            await runtime.concurrency_governor.release(route.name, foreground=foreground)
            capacity_acquired = False

            response_headers = dict(upstream.headers)
            response_headers.update(decision_headers(decision))
            response_headers.update(shaping_headers)
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=response_headers.get("content-type"),
            )

        except UpstreamFailure as exc:
            # Release capacity for this endpoint if acquired
            if capacity_acquired and route is not None:
                await runtime.concurrency_governor.release(route.name, foreground=foreground)
                capacity_acquired = False

            # Record the failure for the circuit breaker of the specific endpoint
            failed_endpoint_name = (
                route.name if route is not None else (exc.endpoint or metadata.endpoint_name)
            )
            if failed_endpoint_name:
                runtime.endpoint_registry.record_upstream_failure(failed_endpoint_name)

            # Check if we can retry on another endpoint
            if can_retry and exc.retryable and route is not None:
                next_route = runtime.endpoint_registry.select_compatible(
                    family=family,
                    model=metadata.model,
                    requires_streaming=requires_streaming,
                    exclude=failed_endpoints | {route.name},
                )
                if next_route is None:
                    fallback = runtime.upstream.route(None)
                    allowed = runtime.endpoint_registry.allows_traffic(fallback.name)
                    if (fallback.name not in (failed_endpoints | {route.name})) and allowed:
                        next_route = fallback

                if next_route is not None:
                    failed_endpoints.add(route.name)
                    await runtime.event_store.append(
                        "upstream.failover",
                        correlation_id=metadata.correlation_id,
                        failed_endpoint=route.name,
                        category=exc.category.value,
                        status_code=exc.status_code,
                    )
                    continue

            # Otherwise, log error event and return error response
            await runtime.event_store.append(
                "upstream.failure",
                correlation_id=metadata.correlation_id,
                category=exc.category.value,
                status_code=exc.status_code,
                endpoint=failed_endpoint_name or "none",
                retryable=exc.retryable,
                path=upstream_path,
            )
            return error_response(
                status_code=exc.status_code,
                message=exc.message,
                error_type=f"computecop_upstream_{exc.category.value}",
                correlation_id=metadata.correlation_id,
                retry_after_seconds=(
                    runtime.config.queue.background_retry_after_seconds if exc.retryable else None
                ),
                extra={"upstream_failure": exc.to_dict()},
            )
        finally:
            if capacity_acquired and route is not None:
                await runtime.concurrency_governor.release(route.name, foreground=foreground)


async def _json_body(request: Request) -> Any:
    try:
        return await request.json()
    except ValueError:
        return None


def _shape_body(
    family: str,
    body: dict[str, Any],
    budget,
) -> tuple[dict[str, Any], dict[str, str]]:
    if family == "ollama":
        return _shape_ollama_body(body, budget)
    if family == "llama_cpp":
        return _shape_llama_cpp_body(body, budget)
    return _shape_openai_body(body, budget)


def _shape_openai_body(
    body: dict[str, Any],
    budget,
) -> tuple[dict[str, Any], dict[str, str]]:
    shaped = dict(body)
    shaping_headers = {"x-computecop-budget-shaped": "false"}

    original_max = None
    for key in ("max_tokens", "max_completion_tokens"):
        if key in shaped:
            try:
                original_max = int(shaped[key])
            except (ValueError, TypeError):
                pass
            if original_max is not None:
                shaped[key] = min(original_max, budget.max_output_tokens)
            else:
                shaped[key] = budget.max_output_tokens

    if "max_tokens" not in shaped and "max_completion_tokens" not in shaped:
        shaped["max_tokens"] = budget.max_output_tokens

    if original_max is not None and original_max > budget.max_output_tokens:
        shaping_headers["x-computecop-budget-shaped"] = "true"
        shaping_headers["x-computecop-original-max-tokens"] = str(original_max)
        shaping_headers["x-computecop-shaped-max-tokens"] = str(budget.max_output_tokens)

    metadata_dict = dict(shaped.get("metadata") or {})
    metadata_dict["computecop_juice_level"] = budget.juice_level
    metadata_dict["computecop_context_budget"] = budget.max_context_tokens
    shaped["metadata"] = metadata_dict

    return shaped, shaping_headers


def _shape_ollama_body(
    body: dict[str, Any],
    budget,
) -> tuple[dict[str, Any], dict[str, str]]:
    shaped = dict(body)
    shaping_headers = {"x-computecop-budget-shaped": "false"}
    options = dict(shaped.get("options") or {})

    original_ctx = None
    if "num_ctx" in options:
        try:
            original_ctx = int(options["num_ctx"])
        except (ValueError, TypeError):
            pass
        if original_ctx is not None:
            options["num_ctx"] = min(original_ctx, budget.max_context_tokens)
        else:
            options["num_ctx"] = budget.max_context_tokens
    else:
        options["num_ctx"] = budget.max_context_tokens

    original_predict = None
    if "num_predict" in options:
        try:
            original_predict = int(options["num_predict"])
        except (ValueError, TypeError):
            pass
        if original_predict is not None:
            options["num_predict"] = min(original_predict, budget.max_output_tokens)
        else:
            options["num_predict"] = budget.max_output_tokens
    else:
        options["num_predict"] = budget.max_output_tokens

    shaped["options"] = options
    shaped["keep_alive"] = shaped.get("keep_alive", "5m" if budget.juice_level >= 50 else "30s")

    is_shaped = False
    if original_ctx is not None and original_ctx > budget.max_context_tokens:
        is_shaped = True
        shaping_headers["x-computecop-original-context-tokens"] = str(original_ctx)
        shaping_headers["x-computecop-shaped-context-tokens"] = str(budget.max_context_tokens)

    if original_predict is not None and original_predict > budget.max_output_tokens:
        is_shaped = True
        shaping_headers["x-computecop-original-max-tokens"] = str(original_predict)
        shaping_headers["x-computecop-shaped-max-tokens"] = str(budget.max_output_tokens)

    if is_shaped:
        shaping_headers["x-computecop-budget-shaped"] = "true"

    return shaped, shaping_headers


def _shape_llama_cpp_body(
    body: dict[str, Any],
    budget,
) -> tuple[dict[str, Any], dict[str, str]]:
    shaped = dict(body)
    shaping_headers = {"x-computecop-budget-shaped": "false"}

    original_max = None
    if "n_predict" in shaped:
        try:
            original_max = int(shaped["n_predict"])
        except (ValueError, TypeError):
            pass
        if original_max is not None:
            shaped["n_predict"] = min(original_max, budget.max_output_tokens)
        else:
            shaped["n_predict"] = budget.max_output_tokens
    elif "max_tokens" in shaped:
        try:
            original_max = int(shaped["max_tokens"])
        except (ValueError, TypeError):
            pass
        if original_max is not None:
            shaped["max_tokens"] = min(original_max, budget.max_output_tokens)
        else:
            shaped["max_tokens"] = budget.max_output_tokens
    else:
        shaped["n_predict"] = budget.max_output_tokens

    original_ctx = None
    if "n_ctx" in shaped:
        try:
            original_ctx = int(shaped["n_ctx"])
        except (ValueError, TypeError):
            pass
        if original_ctx is not None:
            shaped["n_ctx"] = min(original_ctx, budget.max_context_tokens)
        else:
            shaped["n_ctx"] = budget.max_context_tokens
    else:
        shaped["n_ctx"] = budget.max_context_tokens

    cache_prompt = budget.juice_level >= 35
    shaped["cache_prompt"] = bool(shaped.get("cache_prompt", cache_prompt))

    is_shaped = False
    if original_ctx is not None and original_ctx > budget.max_context_tokens:
        is_shaped = True
        shaping_headers["x-computecop-original-context-tokens"] = str(original_ctx)
        shaping_headers["x-computecop-shaped-context-tokens"] = str(budget.max_context_tokens)

    if original_max is not None and original_max > budget.max_output_tokens:
        is_shaped = True
        shaping_headers["x-computecop-original-max-tokens"] = str(original_max)
        shaping_headers["x-computecop-shaped-max-tokens"] = str(budget.max_output_tokens)

    if is_shaped:
        shaping_headers["x-computecop-budget-shaped"] = "true"

    return shaped, shaping_headers


def _select_route(
    runtime: ComputeCopRuntime,
    endpoint_name: str | None,
    family: str,
    *,
    model: str | None = None,
    requires_streaming: bool = False,
    exclude: set[str] | None = None,
):
    if endpoint_name:
        route = runtime.upstream.route(endpoint_name)
        if exclude and route.name in exclude:
            raise UpstreamFailure(
                f"explicit endpoint '{route.name}' failed and is excluded",
                category=UpstreamFailureCategory.UNREACHABLE,
                status_code=503,
                endpoint=route.name,
                retryable=True,
            )
        if not runtime.endpoint_registry.allows_traffic(route.name):
            raise UpstreamFailure(
                f"endpoint '{route.name}' circuit breaker is open",
                category=UpstreamFailureCategory.UNREACHABLE,
                status_code=503,
                endpoint=route.name,
                retryable=True,
                remediation=(
                    f"wait for endpoint '{route.name}' to recover or inspect "
                    "GET /endpoints for circuit breaker status"
                ),
            )
        target_kind = resolve_api_family(family)
        if target_kind is not None and route.kind != target_kind:
            raise UpstreamFailure(
                f"explicit endpoint '{route.name}' of kind '{route.kind.value}' "
                f"is not compatible with requested family '{family}'",
                category=UpstreamFailureCategory.MISCONFIGURED_ENDPOINT,
                status_code=400,
                endpoint=route.name,
                retryable=False,
            )
        if requires_streaming and not route.supports_streaming:
            raise UpstreamFailure(
                f"explicit endpoint '{route.name}' does not support streaming",
                category=UpstreamFailureCategory.MISCONFIGURED_ENDPOINT,
                status_code=400,
                endpoint=route.name,
                retryable=False,
            )
        if model and not runtime.state.residency_tracker.is_model_compatible(model, route.name):
            raise UpstreamFailure(
                f"explicit endpoint '{route.name}' does not support model '{model}'",
                category=UpstreamFailureCategory.MISCONFIGURED_ENDPOINT,
                status_code=400,
                endpoint=route.name,
                retryable=False,
            )
        return route

    # If no explicit endpoint name, use compatibility-based routing
    selected = runtime.endpoint_registry.select_compatible(
        family=family,
        model=model,
        requires_streaming=requires_streaming,
        exclude=exclude,
    )
    if selected is not None:
        return selected

    # Fallback to default route if allowed traffic and not excluded
    fallback = runtime.upstream.route(None)
    allowed = runtime.endpoint_registry.allows_traffic(fallback.name)
    if (not exclude or fallback.name not in exclude) and allowed:
        return fallback

    # Diagnostics to return precise errors
    target_kind = resolve_api_family(family)
    if target_kind is None:
        raise UpstreamFailure(
            f"unknown API family '{family}'",
            category=UpstreamFailureCategory.ROUTE_NOT_FOUND,
            status_code=400,
            endpoint=None,
            retryable=False,
        )

    all_routes = list(runtime.upstream.routes.values())
    if exclude:
        all_routes = [r for r in all_routes if r.name not in exclude]
        if not all_routes:
            raise UpstreamFailure(
                "all configured endpoints are excluded after failures",
                category=UpstreamFailureCategory.UNREACHABLE,
                status_code=503,
                endpoint=None,
                retryable=True,
            )

    family_routes = [r for r in all_routes if r.kind == target_kind]
    if not family_routes:
        known_families = {r.kind.value for r in all_routes}
        kinds_str = ", ".join(sorted(known_families))
        raise UpstreamFailure(
            f"no configured endpoint supports API family '{family}', configured kinds: {kinds_str}",
            category=UpstreamFailureCategory.ROUTE_NOT_FOUND,
            status_code=400,
            endpoint=None,
            retryable=False,
            remediation="configure an endpoint with the requested family in pyproject.toml",
        )

    stream_routes = [r for r in family_routes if not requires_streaming or r.supports_streaming]
    if not stream_routes:
        raise UpstreamFailure(
            f"no endpoint for family '{family}' supports streaming requests",
            category=UpstreamFailureCategory.MISCONFIGURED_ENDPOINT,
            status_code=400,
            endpoint=None,
            retryable=False,
        )

    model_routes = stream_routes
    if model:
        model_routes = [
            r
            for r in stream_routes
            if runtime.state.residency_tracker.is_model_compatible(model, r.name)
        ]
        if not model_routes:
            raise UpstreamFailure(
                f"no compatible endpoint supports model '{model}' for API family '{family}'",
                category=UpstreamFailureCategory.ROUTE_NOT_FOUND,
                status_code=400,
                endpoint=None,
                retryable=False,
            )

    # If we got here, all candidates exist but their circuit breakers are open
    unreachable_names = ", ".join(sorted(r.name for r in model_routes))
    model_str = model or "any"
    raise UpstreamFailure(
        f"all compatible endpoints for family '{family}' and model '{model_str}' "
        f"({unreachable_names}) are currently unreachable or have open circuit breakers",
        category=UpstreamFailureCategory.UNREACHABLE,
        status_code=503,
        endpoint=None,
        retryable=True,
        remediation="inspect GET /endpoints for circuit breaker status and start the local engines",
    )


async def _tracked_stream(
    *,
    runtime: ComputeCopRuntime,
    route_name: str,
    stream: AsyncIterator[bytes],
    endpoint_release: Callable[[], Awaitable[None]] | None = None,
) -> AsyncIterator[bytes]:
    try:
        async for chunk in stream:
            yield chunk
        runtime.endpoint_registry.record_upstream_success(route_name)
    except UpstreamFailure:
        runtime.endpoint_registry.record_upstream_failure(route_name)
        raise
    finally:
        if endpoint_release is not None:
            await endpoint_release()


def json_dumps(value: object) -> str:
    import json

    return json.dumps(value, separators=(",", ":"), sort_keys=True)
