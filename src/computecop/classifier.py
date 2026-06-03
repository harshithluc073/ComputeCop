"""Incoming inference request classification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from computecop.models import RequestClass, RequestMetadata, RequestPriority, new_correlation_id


BACKGROUND_HINT_HEADERS = {
    "x-computecop-background",
    "x-agent-request",
    "x-automation-request",
    "x-background-request",
}

PROMPT_HINT_VALUES = {"prompt", "user_prompt", "foreground", "interactive", "user"}
BACKGROUND_HINT_VALUES = {"request", "background", "background_request", "bulk", "agent", "automation"}


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
        request_class, priority = self._classify(normalized_headers, payload)
        correlation_id = (
            normalized_headers.get("x-correlation-id")
            or normalized_headers.get("x-request-id")
            or new_correlation_id()
        )
        return RequestMetadata(
            method=method.upper(),
            path=path,
            headers=normalized_headers,
            request_class=request_class,
            priority=priority,
            correlation_id=correlation_id,
            client_host=client_host,
            model=_extract_model(payload),
            endpoint_name=normalized_headers.get("x-computecop-endpoint"),
        )

    def _classify(
        self,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
    ) -> tuple[RequestClass, RequestPriority]:
        explicit = _first_value(
            headers.get("x-computecop-class"),
            headers.get("x-computecop-priority"),
            _nested_payload_value(payload, "metadata", "computecop_class"),
            _nested_payload_value(payload, "metadata", "priority"),
            payload.get("computecop_class"),
            payload.get("priority"),
        )
        if explicit in PROMPT_HINT_VALUES:
            return RequestClass.USER_PROMPT, RequestPriority.FOREGROUND
        if explicit in BACKGROUND_HINT_VALUES:
            return RequestClass.BACKGROUND_REQUEST, RequestPriority.BACKGROUND

        if any(_truthy(headers.get(header)) for header in BACKGROUND_HINT_HEADERS):
            return RequestClass.BACKGROUND_REQUEST, RequestPriority.BACKGROUND

        user_agent = headers.get("user-agent", "").casefold()
        if any(token in user_agent for token in ("agent", "automation", "scheduler", "crawler")):
            return RequestClass.BACKGROUND_REQUEST, RequestPriority.BACKGROUND

        if _truthy(_nested_payload_value(payload, "metadata", "interactive")):
            return RequestClass.USER_PROMPT, RequestPriority.INTERACTIVE

        if _truthy(_nested_payload_value(payload, "metadata", "background")):
            return RequestClass.BACKGROUND_REQUEST, RequestPriority.BACKGROUND

        if _looks_like_chat_prompt(payload):
            return RequestClass.USER_PROMPT, RequestPriority.INTERACTIVE

        return RequestClass.BACKGROUND_REQUEST, RequestPriority.BACKGROUND


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
