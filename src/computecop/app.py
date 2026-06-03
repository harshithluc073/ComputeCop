"""FastAPI application factory for the ComputeCop proxy."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from computecop.admission import AdmissionController
from computecop.classifier import RequestClassifier
from computecop.config import RuntimeConfig, cached_config
from computecop.models import to_jsonable
from computecop.offload import OffloadManager
from computecop.policy import JuicePolicyEngine
from computecop.request_queue import AsyncRequestQueue
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
    )


def create_app(config: RuntimeConfig | None = None) -> FastAPI:
    """Create the ComputeCop FastAPI app."""

    runtime = build_runtime(config or cached_config())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.runtime = runtime
        await runtime.telemetry_loop.start()
        try:
            yield
        finally:
            await runtime.telemetry_loop.stop()
            await runtime.queue.close()
            await runtime.upstream.close()

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

    @app.post("/v1/chat/completions")
    async def openai_chat_completions(request: Request) -> Response:
        return await _handle_inference_request(
            runtime=runtime,
            request=request,
            upstream_path="/v1/chat/completions",
            family="openai",
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

    if decision.decision.value in {"reject", "yield"}:
        return _decision_response(decision, status_code=429 if decision.retry_after_seconds else 503)

    route = runtime.upstream.route(metadata.endpoint_name)
    shaped_body = _shape_openai_body(body, decision.budget) if isinstance(body, dict) else body
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
                    "x-computecop-correlation-id": metadata.correlation_id,
                    "x-computecop-decision": decision.decision.value,
                    "x-computecop-juice-level": str(decision.budget.juice_level),
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
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": str(exc),
                    "type": "upstream_error",
                    "correlation_id": metadata.correlation_id,
                }
            },
        )

    response_headers = dict(upstream.headers)
    response_headers["x-computecop-correlation-id"] = metadata.correlation_id
    response_headers["x-computecop-decision"] = decision.decision.value
    response_headers["x-computecop-juice-level"] = str(decision.budget.juice_level)
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


def _decision_response(decision, status_code: int) -> JSONResponse:
    headers = {"x-computecop-correlation-id": decision.correlation_id}
    if decision.retry_after_seconds is not None:
        headers["retry-after"] = str(max(1, int(decision.retry_after_seconds)))
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={
            "error": {
                "message": decision.reason,
                "type": f"computecop_{decision.decision.value}",
                "correlation_id": decision.correlation_id,
                "retry_after_seconds": decision.retry_after_seconds,
                "queue_position": decision.queue_position,
            },
            "decision": to_jsonable(decision),
        },
    )
