from __future__ import annotations

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from computecop.classifier import RequestClassifier
from computecop.config import EndpointConfig
from computecop.models import RequestClass


def test_validate_endpoint_examples() -> None:
    examples_dir = Path(__file__).parent.parent / "examples"
    endpoint_json_path = examples_dir / "endpoints.ollama-and-llama-cpp.json"
    
    assert endpoint_json_path.exists(), f"Missing example: {endpoint_json_path}"
    
    with open(endpoint_json_path, "r", encoding="utf-8") as f:
        endpoints_data = json.load(f)
        
    assert isinstance(endpoints_data, list), "Endpoints example should be a JSON list"
    assert len(endpoints_data) > 0, "Endpoints list should not be empty"
    
    for item in endpoints_data:
        # Validate that the item matches EndpointConfig structure
        try:
            EndpointConfig.model_validate(item)
        except ValidationError as exc:
            pytest.fail(f"Failed to validate endpoint configuration: {item}. Error: {exc}")


def test_validate_request_examples() -> None:
    examples_dir = Path(__file__).parent.parent / "examples"
    classifier = RequestClassifier()
    
    # 1. Llama-cpp background request
    llama_cpp_path = examples_dir / "llama-cpp-background-request.json"
    assert llama_cpp_path.exists()
    with open(llama_cpp_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = classifier.classify(
        method="POST",
        path="/completion",
        headers=data.get("headers", {}),
        body=data.get("body", {}),
    )
    assert meta.request_class == RequestClass.BACKGROUND_REQUEST
    
    # 2. Ollama background request
    ollama_path = examples_dir / "ollama-background-request.json"
    assert ollama_path.exists()
    with open(ollama_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = classifier.classify(
        method="POST",
        path="/api/chat",
        headers=data.get("headers", {}),
        body=data.get("body", {}),
    )
    assert meta.request_class == RequestClass.BACKGROUND_REQUEST
    
    # 3. OpenAI foreground prompt
    openai_path = examples_dir / "openai-foreground-prompt.json"
    assert openai_path.exists()
    with open(openai_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = classifier.classify(
        method="POST",
        path="/v1/chat/completions",
        headers=data.get("headers", {}),
        body=data.get("body", {}),
    )
    assert meta.request_class == RequestClass.USER_PROMPT
