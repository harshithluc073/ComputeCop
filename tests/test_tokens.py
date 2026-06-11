from __future__ import annotations

from computecop.tokens import RequestTokenEstimator


def test_estimator_empty_or_none() -> None:
    estimator = RequestTokenEstimator()
    res = estimator.estimate(None)
    assert res.estimated_tokens == 0
    assert res.confidence == 1.0
    assert res.field_contribution == {}

    res_empty = estimator.estimate({})
    assert res_empty.estimated_tokens == 0
    assert res_empty.confidence == 1.0
    assert res_empty.field_contribution == {}


def test_estimator_prompt_heuristics() -> None:
    estimator = RequestTokenEstimator()
    # "hello" is 5 chars. At ratio=4.0, round(5/4) = 1 token
    res = estimator.estimate({"prompt": "hello"})
    assert res.estimated_tokens == 1
    assert res.confidence == 0.8
    assert res.field_contribution == {"prompt": 1}

    # Custom ratio: 1.0
    res_custom = estimator.estimate({"prompt": "hello"}, chars_per_token_ratio=1.0)
    assert res_custom.estimated_tokens == 5
    assert res_custom.field_contribution == {"prompt": 5}

    # List of prompts
    res_list = estimator.estimate({"prompt": ["hello", " world"]})
    assert res_list.estimated_tokens == 3  # "hello world" is 11 chars. 11/4 = 2.75 -> 3
    assert res_list.field_contribution == {"prompt": 3}

    # System and template
    res_sys = estimator.estimate({"system": "sys", "template": "temp"})
    assert res_sys.estimated_tokens == 2  # "sys" (3) + "temp" (4) = 7. 7/4 = 1.75 -> 2
    assert res_sys.field_contribution == {"prompt": 2}


def test_estimator_messages_heuristics() -> None:
    estimator = RequestTokenEstimator()
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
    }
    # 28 chars + 6 chars = 34 chars. 34/4 = 8.5. round(8.5) = 8.
    res = estimator.estimate(payload)
    assert res.estimated_tokens == 8
    assert res.confidence == 0.8
    assert res.field_contribution == {"messages": 8}


def test_estimator_rich_messages_and_images() -> None:
    estimator = RequestTokenEstimator()
    # List content
    payload_list = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image:"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64..."}},
                ],
            }
        ]
    }
    res = estimator.estimate(payload_list)
    # "Describe this image:" (20 chars) -> 20/4 = 5 tokens. Image adds 500 tokens.
    assert res.estimated_tokens == 505
    assert res.confidence == 0.5
    assert res.field_contribution == {"messages": 5, "other_payloads": 500}

    # Root images
    payload_root_img = {
        "prompt": "describe",
        "images": ["img1", "img2"],
    }
    res_root = estimator.estimate(payload_root_img)
    # "describe" (8 chars) -> 8/4 = 2 tokens. 2 images -> 1000 tokens.
    assert res_root.estimated_tokens == 1002
    assert res_root.confidence == 0.5
    assert res_root.field_contribution == {"prompt": 2, "other_payloads": 1000}


def test_estimator_tools_and_functions() -> None:
    estimator = RequestTokenEstimator()
    payload = {
        "messages": [
            {"role": "user", "content": "Call tool"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "1",
                        "type": "function",
                        "function": {"name": "test", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "tool result content", "tool_call_id": "1"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "test", "description": "test"}}
        ],
    }
    res = estimator.estimate(payload)
    # Should detect tools definitions, tool calls, and responses under tool_payloads
    assert "tool_payloads" in res.field_contribution
    assert res.field_contribution["tool_payloads"] > 0
    assert res.confidence == 0.5
