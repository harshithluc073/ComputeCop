"""FastAPI application factory for the ComputeCop proxy."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from fastapi import FastAPI

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

    return app

app = create_app()
