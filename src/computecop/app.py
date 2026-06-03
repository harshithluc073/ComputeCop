"""FastAPI application factory for the ComputeCop proxy."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from computecop.admission import AdmissionController
from computecop.classifier import RequestClassifier
from computecop.config import RuntimeConfig, cached_config
from computecop.events import JsonlEventStore
from computecop.models import EndpointKind, to_jsonable
from computecop.offload import OffloadManager
from computecop.policy import JuicePolicyEngine
from computecop.request_queue import AsyncRequestQueue
from computecop.responses import decision_headers, decision_response, error_response
from computecop.state import RuntimeStateStore
from computecop.telemetry import PsutilTelemetrySampler
from computecop.telemetry_loop import TelemetryLoop
from computecop.thermal import ThermalDetector, ThermalThresholds
from computecop.upstream import UpstreamRouter
from computecop.upstream import UpstreamError
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
    upstream: UpstreamRouter
    telemetry_loop: TelemetryLoop
    yield_controller: RamYieldController
    offload_manager: OffloadManager
    event_store: JsonlEventStore
    queue_workers: list[asyncio.Task[None]]

    async def start(self) -> None:
        """Start background runtime services."""

        await self.telemetry_loop.start()
        if not self.queue_workers:
            for index in range(self.config.policy.max_background_concurrency):
                self.queue_workers.append(
                    asyncio.create_task(
                        self.queue.run_worker(),
                        name=f"computecop-queue-worker-{index}",
                    )
                )

    async def stop(self) -> None:
        """Stop background runtime services."""

        await self.telemetry_loop.stop()
        await self.queue.close()
        for worker in self.queue_workers:
            worker.cancel()
        for worker in self.queue_workers:
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await worker
        self.queue_workers.clear()
        await self.upstream.close()


def build_runtime(config: RuntimeConfig) -> ComputeCopRuntime:
    """Build the runtime dependency graph."""

    thresholds = ThermalThresholds(
        warm_celsius=config.policy.thermal_warm_celsius,
        hot_celsius=config.policy.thermal_hot_celsius,
        critical_celsius=config.policy.thermal_critical_celsius,
    )
    sampler = PsutilTelemetrySampler(thermal_detector=ThermalDetector(thresholds))
    state = RuntimeStateStore()
    policy = JuicePolicyEngine(config.policy)
    routes = [endpoint.to_route() for endpoint in config.endpoints]
    yield_controller = RamYieldController(config.policy)
    offload_manager = OffloadManager(routes)
    event_store = JsonlEventStore(config.event_log_path)
    yield_controller.register_offload_hook(offload_manager.offload_all)
    telemetry_loop = TelemetryLoop(
        sampler=sampler,
        interval_seconds=config.telemetry.interval_seconds,
        smoothing_window=config.telemetry.smoothing_window,
    )

    async def update_runtime(sample):
        await state.update_telemetry(sample)
        await yield_controller.update(sample)
        pressure = policy.evaluate(sample)
        await state.set_policy_state(
            system_state=pressure.system_state,
            global_juice_level=pressure.global_juice_level,
            yield_active=pressure.yield_active,
            yield_reason=pressure.yield_reason,
        )
        if pressure.yield_active:
            await event_store.append("policy.yield", reason=pressure.yield_reason)

    telemetry_loop.subscribe(update_runtime)

    return ComputeCopRuntime(
        config=config,
        state=state,
        classifier=RequestClassifier(),
        policy=policy,
        admission=AdmissionController(policy, config.queue),
        queue=AsyncRequestQueue(config.queue),
        upstream=UpstreamRouter(routes),
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
        version="0.1.0",
        description="Local inference traffic controller with telemetry-aware compute budgeting.",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.get("/health")
    async def health() -> dict[str, object]:
        probe = await runtime.upstream.probe()
        return {
            "ok": True,
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
    pressure = runtime.policy.evaluate(snapshot.telemetry)
    decision = runtime.admission.decide(
        metadata,
        pressure,
        queue_size=runtime.queue.counters().queued,
    )
    await runtime.state.record_decision(decision)
    await runtime.event_store.append(
        "admission.decision",
        decision=decision,
        path=metadata.path,
        model=metadata.model,
        endpoint=metadata.endpoint_name,
    )

    if decision.decision.value in {"reject", "yield"}:
        return decision_response(decision, status_code=429 if decision.retry_after_seconds else 503)

    route = _select_route(runtime, metadata.endpoint_name, family)
    shaped_body = _shape_body(family, body, decision.budget) if isinstance(body, dict) else body
    headers = dict(request.headers)
    headers["x-computecop-correlation-id"] = metadata.correlation_id
    headers["x-computecop-juice-level"] = str(decision.budget.juice_level)

    try:
        if isinstance(shaped_body, dict) and shaped_body.get("stream") is True and route.supports_streaming:
            return StreamingResponse(
                runtime.upstream.stream(
                    route,
                    method=request.method,
                    path=upstream_path,
                    headers=headers,
                    json_body=shaped_body,
                ),
                media_type="text/event-stream",
                headers={
                    **decision_headers(decision),
                },
            )
        upstream = await runtime.upstream.request(
            route,
            method=request.method,
            path=upstream_path,
            headers=headers,
            json_body=shaped_body,
        )
    except UpstreamError as exc:
        return error_response(
            status_code=exc.status_code,
            message=str(exc),
            error_type="upstream_error",
            correlation_id=metadata.correlation_id,
        )

    response_headers = dict(upstream.headers)
    response_headers.update(decision_headers(decision))
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=response_headers.get("content-type"),
    )


async def _json_body(request: Request) -> Any:
    try:
        return await request.json()
    except ValueError:
        return None


def _shape_body(family: str, body: dict[str, Any], budget) -> dict[str, Any]:
    if family == "ollama":
        return _shape_ollama_body(body, budget)
    if family == "llama_cpp":
        return _shape_llama_cpp_body(body, budget)
    return _shape_openai_body(body, budget)


def _shape_openai_body(body: dict[str, Any], budget) -> dict[str, Any]:
    shaped = dict(body)
    for key in ("max_tokens", "max_completion_tokens"):
        if key in shaped:
            shaped[key] = min(int(shaped[key]), budget.max_output_tokens)
    if "max_tokens" not in shaped and "max_completion_tokens" not in shaped:
        shaped["max_tokens"] = budget.max_output_tokens
    metadata = dict(shaped.get("metadata") or {})
    metadata["computecop_juice_level"] = budget.juice_level
    metadata["computecop_context_budget"] = budget.max_context_tokens
    shaped["metadata"] = metadata
    return shaped


def _shape_ollama_body(body: dict[str, Any], budget) -> dict[str, Any]:
    shaped = dict(body)
    options = dict(shaped.get("options") or {})
    if "num_ctx" in options:
        options["num_ctx"] = min(int(options["num_ctx"]), budget.max_context_tokens)
    else:
        options["num_ctx"] = budget.max_context_tokens
    if "num_predict" in options:
        options["num_predict"] = min(int(options["num_predict"]), budget.max_output_tokens)
    else:
        options["num_predict"] = budget.max_output_tokens
    shaped["options"] = options
    shaped["keep_alive"] = shaped.get("keep_alive", "5m" if budget.juice_level >= 50 else "30s")
    return shaped


def _shape_llama_cpp_body(body: dict[str, Any], budget) -> dict[str, Any]:
    shaped = dict(body)
    if "n_predict" in shaped:
        shaped["n_predict"] = min(int(shaped["n_predict"]), budget.max_output_tokens)
    elif "max_tokens" in shaped:
        shaped["max_tokens"] = min(int(shaped["max_tokens"]), budget.max_output_tokens)
    else:
        shaped["n_predict"] = budget.max_output_tokens

    if "n_ctx" in shaped:
        shaped["n_ctx"] = min(int(shaped["n_ctx"]), budget.max_context_tokens)
    else:
        shaped["n_ctx"] = budget.max_context_tokens

    cache_prompt = budget.juice_level >= 35
    shaped["cache_prompt"] = bool(shaped.get("cache_prompt", cache_prompt))
    return shaped


def _select_route(runtime: ComputeCopRuntime, endpoint_name: str | None, family: str):
    if endpoint_name:
        return runtime.upstream.route(endpoint_name)
    preferred = {
        "ollama": EndpointKind.OLLAMA,
        "llama_cpp": EndpointKind.LLAMA_CPP,
        "openai": EndpointKind.OPENAI_COMPATIBLE,
    }.get(family)
    if preferred is not None:
        for route in runtime.upstream.routes.values():
            if route.kind == preferred:
                return route
    return runtime.upstream.route(None)
