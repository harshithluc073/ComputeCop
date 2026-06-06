"""Incoming inference request classification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from computecop.models import (
    ClassificationResult,
    RequestClass,
    RequestMetadata,
    RequestPriority,
    new_correlation_id,
)

BACKGROUND_HINT_HEADERS = {
    "x-computecop-background",
    "x-agent-request",
    "x-automation-request",
    "x-background-request",
}

PROMPT_HINT_VALUES = {"prompt", "user_prompt", "foreground", "interactive", "user"}
BACKGROUND_HINT_VALUES = {
    "request",
    "background",
    "background_request",
    "bulk",
    "agent",
    "automation",
}


class RequestClassifier:
    """Classify proxy requests into foreground prompts and background requests."""

    def classify(
        self,
        *,
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: Mapping[str, Any] | None = None,
        client_host: str | None = None,
    ) -> RequestMetadata:
        normalized_headers = {key.lower(): str(value) for key, value in headers.items()}
        payload = body or {}
        result = self._classify_rich(normalized_headers, payload)
        correlation_id = (
            normalized_headers.get("x-correlation-id")
            or normalized_headers.get("x-request-id")
            or new_correlation_id()
        )
        return RequestMetadata(
            method=method.upper(),
            path=path,
            headers=normalized_headers,
            request_class=result.request_class,
            priority=result.priority,
            correlation_id=correlation_id,
            client_host=client_host,
            model=_extract_model(payload),
            endpoint_name=normalized_headers.get("x-computecop-endpoint"),
            classification=result,
        )

    def _classify(
        self,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> tuple[RequestClass, RequestPriority]:
        res = self._classify_rich(headers, payload)
        return res.request_class, res.priority

    def _classify_rich(
        self,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> ClassificationResult:
        header_class = headers.get("x-computecop-class")
        header_priority = headers.get("x-computecop-priority")
        payload_meta_class = _nested_payload_value(payload, "metadata", "computecop_class")
        payload_meta_priority = _nested_payload_value(payload, "metadata", "priority")
        payload_root_class = payload.get("computecop_class")
        payload_root_priority = payload.get("priority")

        explicit = _first_value(
            header_class,
            header_priority,
            payload_meta_class,
            payload_meta_priority,
            payload_root_class,
            payload_root_priority,
        )

        if explicit is not None:
            if explicit in PROMPT_HINT_VALUES:
                is_header = _first_value(header_class, header_priority) is not None
                return ClassificationResult(
                    request_class=RequestClass.USER_PROMPT,
                    priority=RequestPriority.FOREGROUND,
                    confidence_score=1.0 if is_header else 0.5,
                    matched_signals=("explicit_header" if is_header else "explicit_payload_field",),
                )
            if explicit in BACKGROUND_HINT_VALUES:
                is_header = _first_value(header_class, header_priority) is not None
                return ClassificationResult(
                    request_class=RequestClass.BACKGROUND_REQUEST,
                    priority=RequestPriority.BACKGROUND,
                    confidence_score=1.0 if is_header else 0.5,
                    matched_signals=("explicit_header" if is_header else "explicit_payload_field",),
                )

        for header in BACKGROUND_HINT_HEADERS:
            if _truthy(headers.get(header)):
                return ClassificationResult(
                    request_class=RequestClass.BACKGROUND_REQUEST,
                    priority=RequestPriority.BACKGROUND,
                    confidence_score=1.0,
                    matched_signals=(f"header:{header}",),
                )

        user_agent = headers.get("user-agent", "").casefold()
        for token in ("agent", "automation", "scheduler", "crawler"):
            if token in user_agent:
                return ClassificationResult(
                    request_class=RequestClass.BACKGROUND_REQUEST,
                    priority=RequestPriority.BACKGROUND,
                    confidence_score=0.5,
                    matched_signals=(f"user_agent:{token}",),
                )

        if _truthy(_nested_payload_value(payload, "metadata", "interactive")):
            return ClassificationResult(
                request_class=RequestClass.USER_PROMPT,
                priority=RequestPriority.INTERACTIVE,
                confidence_score=0.5,
                matched_signals=("payload_interactive_flag",),
            )

        if _truthy(_nested_payload_value(payload, "metadata", "background")):
            return ClassificationResult(
                request_class=RequestClass.BACKGROUND_REQUEST,
                priority=RequestPriority.BACKGROUND,
                confidence_score=0.5,
                matched_signals=("payload_background_flag",),
            )

        if _looks_like_chat_prompt(payload):
            return ClassificationResult(
                request_class=RequestClass.USER_PROMPT,
                priority=RequestPriority.INTERACTIVE,
                confidence_score=0.5,
                matched_signals=("chat_prompt_heuristic",),
            )

        ambiguous_signals = []
        if explicit is not None:
            is_header = _first_value(header_class, header_priority) is not None
            if is_header:
                ambiguous_signals.append(f"unrecognized_explicit_header:{explicit}")
            else:
                ambiguous_signals.append(f"unrecognized_explicit_payload:{explicit}")

        return ClassificationResult(
            request_class=RequestClass.BACKGROUND_REQUEST,
            priority=RequestPriority.BACKGROUND,
            confidence_score=0.1,
            matched_signals=(),
            ambiguous_signals=tuple(ambiguous_signals),
            recommended_header_fixes=("add x-computecop-background: true for automated work",),
        )


def _first_value(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        cleaned = str(value).strip().casefold()
        if cleaned:
            return cleaned
    return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().casefold() in {"1", "true", "yes", "on", "y"}


def _nested_payload_value(payload: Mapping[str, Any], section: str, key: str) -> object | None:
    value = payload.get(section)
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _extract_model(payload: Mapping[str, Any]) -> str | None:
    model = payload.get("model")
    return str(model) if model is not None else None


def _looks_like_chat_prompt(payload: Mapping[str, Any]) -> bool:
    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, Mapping) and last.get("role") == "user":
            return True
    prompt = payload.get("prompt")
    return isinstance(prompt, str) and prompt.strip() != ""
