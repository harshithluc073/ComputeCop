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

