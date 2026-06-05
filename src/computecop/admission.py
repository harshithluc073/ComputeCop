"""Admission control for incoming proxy requests."""

from __future__ import annotations

from computecop.config import QueueConfig
from computecop.models import (
    AdmissionDecision,
    DecisionType,
    RequestClass,
    RequestMetadata,
    RequestPriority,
    SystemState,
)
from computecop.policy import JuicePolicyEngine, PressureReport


class AdmissionController:
    """Apply policy decisions to request metadata."""

    def __init__(self, policy: JuicePolicyEngine, queue_config: QueueConfig) -> None:
        self.policy = policy
        self.queue_config = queue_config

    def decide(
        self,
        metadata: RequestMetadata,
        pressure: PressureReport,
        *,
        queue_size: int,
    ) -> AdmissionDecision:
        """Return an admission decision for a request."""

        budget = self.policy.budget_for(metadata.request_class, pressure)
        base_trace = pressure.trace

        if metadata.request_class == RequestClass.USER_PROMPT:
            decision = DecisionType.ALLOW
            return AdmissionDecision(
                decision=decision,
                request_class=metadata.request_class,
                priority=metadata.priority,
                budget=budget,
                reason="foreground prompt admitted",
                correlation_id=metadata.correlation_id,
                trace=base_trace.with_admission(
                    request_class=metadata.request_class,
                    priority=metadata.priority,
                    decision=decision,
                    endpoint_name=metadata.endpoint_name,
                    path=metadata.path,
                    queue_size=queue_size,
                    queue_position=None,
                    budget=budget,
                    summary="foreground prompt admitted",
                ),
            )

        if queue_size >= self.queue_config.max_size:
            decision = DecisionType.REJECT
            return AdmissionDecision(
                decision=decision,
                request_class=metadata.request_class,
                priority=metadata.priority,
                budget=budget,
                reason="background queue is full",
                correlation_id=metadata.correlation_id,
                retry_after_seconds=self.queue_config.background_retry_after_seconds,
                trace=base_trace.with_admission(
                    request_class=metadata.request_class,
                    priority=metadata.priority,
                    decision=decision,
                    endpoint_name=metadata.endpoint_name,
                    path=metadata.path,
                    queue_size=queue_size,
                    queue_position=None,
                    budget=budget,
                    summary="background queue is full",
                ),
            )

        if pressure.yield_active:
            decision = DecisionType.YIELD
            return AdmissionDecision(
                decision=decision,
                request_class=metadata.request_class,
                priority=metadata.priority,
                budget=budget,
                reason=pressure.yield_reason or "system is yielding resources",
                correlation_id=metadata.correlation_id,
                retry_after_seconds=self.queue_config.background_retry_after_seconds,
                queue_position=queue_size + 1,
                trace=base_trace.with_admission(
                    request_class=metadata.request_class,
                    priority=metadata.priority,
                    decision=decision,
                    endpoint_name=metadata.endpoint_name,
                    path=metadata.path,
                    queue_size=queue_size,
                    queue_position=queue_size + 1,
                    budget=budget,
                    summary=pressure.yield_reason or "system is yielding resources",
                ),
            )

        if pressure.system_state in {SystemState.PRESSURED, SystemState.RECOVERING}:
            decision = DecisionType.THROTTLE
            return AdmissionDecision(
                decision=decision,
                request_class=metadata.request_class,
                priority=RequestPriority.BACKGROUND,
                budget=budget,
                reason=budget.reason,
                correlation_id=metadata.correlation_id,
                queue_position=queue_size + 1,
                trace=base_trace.with_admission(
                    request_class=metadata.request_class,
                    priority=RequestPriority.BACKGROUND,
                    decision=decision,
                    endpoint_name=metadata.endpoint_name,
                    path=metadata.path,
                    queue_size=queue_size,
                    queue_position=queue_size + 1,
                    budget=budget,
                    summary=budget.reason,
                ),
            )

        decision = DecisionType.ALLOW
        return AdmissionDecision(
            decision=decision,
            request_class=metadata.request_class,
            priority=metadata.priority,
            budget=budget,
            reason="background request admitted",
            correlation_id=metadata.correlation_id,
            trace=base_trace.with_admission(
                request_class=metadata.request_class,
                priority=metadata.priority,
                decision=decision,
                endpoint_name=metadata.endpoint_name,
                path=metadata.path,
                queue_size=queue_size,
                queue_position=None,
                budget=budget,
                summary="background request admitted",
            ),
        )
