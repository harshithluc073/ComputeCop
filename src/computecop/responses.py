"""HTTP response contracts for ComputeCop."""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

from computecop.models import AdmissionDecision, to_jsonable


def decision_headers(decision: AdmissionDecision) -> dict[str, str]:
    """Return standard ComputeCop decision headers."""

    headers = {
        "x-computecop-correlation-id": decision.correlation_id,
        "x-computecop-decision": decision.decision.value,
        "x-computecop-request-class": decision.request_class.value,
        "x-computecop-priority": decision.priority.value,
        "x-computecop-juice-level": str(decision.budget.juice_level),
    }
    if decision.retry_after_seconds is not None:
        headers["retry-after"] = str(max(1, int(decision.retry_after_seconds)))
    return headers


def error_response(
    *,
    status_code: int,
    message: str,
    error_type: str,
    correlation_id: str,
    retry_after_seconds: float | None = None,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    """Return a normalized ComputeCop JSON error response."""

    headers = {"x-computecop-correlation-id": correlation_id}
    if retry_after_seconds is not None:
        headers["retry-after"] = str(max(1, int(retry_after_seconds)))
    payload: dict[str, Any] = {
        "error": {
            "message": message,
            "type": error_type,
            "correlation_id": correlation_id,
            "retry_after_seconds": retry_after_seconds,
        }
    }
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=status_code, headers=headers, content=payload)


def decision_response(decision: AdmissionDecision, status_code: int) -> JSONResponse:
    """Return a normalized response for admission denials or yield decisions."""

    return error_response(
        status_code=status_code,
        message=decision.reason,
        error_type=f"computecop_{decision.decision.value}",
        correlation_id=decision.correlation_id,
        retry_after_seconds=decision.retry_after_seconds,
        extra={
            "queue_position": decision.queue_position,
            "decision": to_jsonable(decision),
        },
    )
