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

        if metadata.request_class == RequestClass.USER_PROMPT:
            return AdmissionDecision(
                decision=DecisionType.ALLOW,
                request_class=metadata.request_class,
                priority=metadata.priority,
                budget=budget,
                reason="foreground prompt admitted",
                correlation_id=metadata.correlation_id,
            )

        if queue_size >= self.queue_config.max_size:
            return AdmissionDecision(
                decision=DecisionType.REJECT,
                request_class=metadata.request_class,
                priority=metadata.priority,
                budget=budget,
                reason="background queue is full",
                correlation_id=metadata.correlation_id,
                retry_after_seconds=self.queue_config.background_retry_after_seconds,
            )

        if pressure.yield_active:
            return AdmissionDecision(
                decision=DecisionType.YIELD,
                request_class=metadata.request_class,
                priority=metadata.priority,
                budget=budget,
                reason=pressure.yield_reason or "system is yielding resources",
                correlation_id=metadata.correlation_id,
                retry_after_seconds=self.queue_config.background_retry_after_seconds,
                queue_position=queue_size + 1,
            )

        if pressure.system_state in {SystemState.PRESSURED, SystemState.RECOVERING}:
            return AdmissionDecision(
                decision=DecisionType.THROTTLE,
                request_class=metadata.request_class,
                priority=RequestPriority.BACKGROUND,
                budget=budget,
                reason=budget.reason,
                correlation_id=metadata.correlation_id,
                queue_position=queue_size + 1,
            )

        return AdmissionDecision(
            decision=DecisionType.ALLOW,
            request_class=metadata.request_class,
            priority=metadata.priority,
            budget=budget,
            reason="background request admitted",
            correlation_id=metadata.correlation_id,
        )
