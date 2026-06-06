from __future__ import annotations

from computecop.classifier import RequestClassifier
from computecop.models import RequestClass, RequestPriority


def test_explicit_prompt_header_wins() -> None:
    metadata = RequestClassifier().classify(
        method="post",
        path="/v1/chat/completions",
        headers={"x-computecop-class": "prompt"},
        body={"metadata": {"background": True}},
    )
    assert metadata.request_class == RequestClass.USER_PROMPT
    assert metadata.priority == RequestPriority.FOREGROUND


def test_background_header_classifies_request() -> None:
    metadata = RequestClassifier().classify(
        method="post",
        path="/api/chat",
        headers={"x-computecop-background": "true"},
        body={"prompt": "summarize logs"},
    )
    assert metadata.request_class == RequestClass.BACKGROUND_REQUEST


def test_classification_result_details() -> None:
    # 1. High confidence explicit header
    metadata = RequestClassifier().classify(
        method="post",
        path="/api/chat",
        headers={"x-computecop-class": "prompt"},
    )
    res = metadata.classification
    assert res is not None
    assert res.request_class == RequestClass.USER_PROMPT
    assert res.priority == RequestPriority.FOREGROUND
    assert res.confidence_score == 1.0
    assert "explicit_header" in res.matched_signals

    # 2. High confidence background header
    metadata_bg = RequestClassifier().classify(
        method="post",
        path="/api/chat",
        headers={"x-computecop-background": "true"},
    )
    res_bg = metadata_bg.classification
    assert res_bg is not None
    assert res_bg.request_class == RequestClass.BACKGROUND_REQUEST
    assert res_bg.confidence_score == 1.0
    assert "header:x-computecop-background" in res_bg.matched_signals

    # 3. Medium confidence payload field
    metadata_payload = RequestClassifier().classify(
        method="post",
        path="/api/chat",
        headers={},
        body={"computecop_class": "prompt"},
    )
    res_payload = metadata_payload.classification
    assert res_payload is not None
    assert res_payload.request_class == RequestClass.USER_PROMPT
    assert res_payload.confidence_score == 0.5
    assert "explicit_payload_field" in res_payload.matched_signals

    # 4. Medium confidence User-Agent
    metadata_ua = RequestClassifier().classify(
        method="post",
        path="/api/chat",
        headers={"user-agent": "ComputeCopAutomationAgent/1.0"},
    )
    res_ua = metadata_ua.classification
    assert res_ua is not None
    assert res_ua.request_class == RequestClass.BACKGROUND_REQUEST
    assert res_ua.confidence_score == 0.5
    assert "user_agent:agent" in res_ua.matched_signals

    # 5. Medium confidence chat prompt heuristic
    metadata_chat = RequestClassifier().classify(
        method="post",
        path="/api/chat",
        headers={},
        body={"messages": [{"role": "user", "content": "hello"}]},
    )
    res_chat = metadata_chat.classification
    assert res_chat is not None
    assert res_chat.request_class == RequestClass.USER_PROMPT
    assert res_chat.confidence_score == 0.5
    assert "chat_prompt_heuristic" in res_chat.matched_signals

    # 6. Low confidence fallback
    metadata_fallback = RequestClassifier().classify(
        method="post",
        path="/api/chat",
        headers={},
        body={},
    )
    res_fallback = metadata_fallback.classification
    assert res_fallback is not None
    assert res_fallback.request_class == RequestClass.BACKGROUND_REQUEST
    assert res_fallback.confidence_score == 0.1
    assert len(res_fallback.matched_signals) == 0
    assert (
        "add x-computecop-background: true for automated work"
        in res_fallback.recommended_header_fixes
    )

    # 7. Low confidence fallback with unrecognized signals (ambiguous)
    metadata_ambig = RequestClassifier().classify(
        method="post",
        path="/api/chat",
        headers={"x-computecop-class": "invalid-class-name"},
    )
    res_ambig = metadata_ambig.classification
    assert res_ambig is not None
    assert res_ambig.confidence_score == 0.1
    assert "unrecognized_explicit_header:invalid-class-name" in res_ambig.ambiguous_signals
