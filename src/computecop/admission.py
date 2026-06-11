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

        estimated_tokens = (
            metadata.token_estimation.estimated_tokens if metadata.token_estimation else None
        )
        confidence = metadata.token_estimation.confidence if metadata.token_estimation else None
        original_ctx = metadata.original_context_tokens
        original_max = metadata.original_max_tokens

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
                    estimated_prompt_tokens=estimated_tokens,
                    estimated_prompt_confidence=confidence,
                    original_context_tokens=original_ctx,
                    original_max_tokens=original_max,
                ),
                classification=metadata.classification,
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
                    estimated_prompt_tokens=estimated_tokens,
                    estimated_prompt_confidence=confidence,
                    original_context_tokens=original_ctx,
                    original_max_tokens=original_max,
                ),
                classification=metadata.classification,
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
                    estimated_prompt_tokens=estimated_tokens,
                    estimated_prompt_confidence=confidence,
                    original_context_tokens=original_ctx,
                    original_max_tokens=original_max,
                ),
                classification=metadata.classification,
            )

        is_too_large = (
            metadata.request_class == RequestClass.BACKGROUND_REQUEST
            and estimated_tokens is not None
            and estimated_tokens > budget.max_context_tokens
        )

        if pressure.system_state in {SystemState.PRESSURED, SystemState.RECOVERING} or is_too_large:
            decision = DecisionType.THROTTLE
            reason = (
                f"large background request ({estimated_tokens} tokens) "
                f"exceeds context budget ({budget.max_context_tokens} tokens)"
                if is_too_large
                else budget.reason
            )
            return AdmissionDecision(
                decision=decision,
                request_class=metadata.request_class,
                priority=RequestPriority.BACKGROUND,
                budget=budget,
                reason=reason,
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
                    summary=reason,
                    estimated_prompt_tokens=estimated_tokens,
                    estimated_prompt_confidence=confidence,
                    original_context_tokens=original_ctx,
                    original_max_tokens=original_max,
                ),
                classification=metadata.classification,
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
                estimated_prompt_tokens=estimated_tokens,
                estimated_prompt_confidence=confidence,
                original_context_tokens=original_ctx,
                original_max_tokens=original_max,
            ),
            classification=metadata.classification,
        )
