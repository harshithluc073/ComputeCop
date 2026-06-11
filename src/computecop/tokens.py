"""Dependency-light token estimation heuristics for ComputeCop."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from computecop.models import TokenEstimationResult


class RequestTokenEstimator:
    """Estimates the input token count of incoming inference request payloads."""

    def estimate(
        self,
        body: Mapping[str, Any] | None,
        chars_per_token_ratio: float = 4.0,
    ) -> TokenEstimationResult:
        """Estimate the request tokens and return breakdown and confidence."""

        if not body or not isinstance(body, Mapping):
            return TokenEstimationResult(
                estimated_tokens=0,
                confidence=1.0,
                field_contribution={},
            )

        chars_per_token = max(1.0, chars_per_token_ratio)
        has_rich_data = False
        contributions: dict[str, int] = {}

        # 1. Prompt string (OpenAI completions, Ollama generate, llama.cpp completion)
        prompt_chars = 0
        prompt_val = body.get("prompt")
        if isinstance(prompt_val, str):
            prompt_chars += len(prompt_val)
        elif isinstance(prompt_val, list):
            for item in prompt_val:
                if isinstance(item, str):
                    prompt_chars += len(item)

        # Ollama system and template fields
        for field_name in ("system", "template"):
            val = body.get(field_name)
            if isinstance(val, str):
                prompt_chars += len(val)

        if prompt_chars > 0:
            contributions["prompt"] = math.ceil(prompt_chars / chars_per_token)

        # 2. Messages (OpenAI chat completions, Ollama chat, llama.cpp chat completions)
        messages_chars = 0
        tool_chars = 0
        image_tokens = 0

        messages_val = body.get("messages")
        if isinstance(messages_val, list):
            for msg in messages_val:
                if not isinstance(msg, Mapping):
                    continue

                role = str(msg.get("role", "")).casefold()

                # Content parsing
                content = msg.get("content")
                msg_content_chars = 0
                if isinstance(content, str):
                    msg_content_chars += len(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, Mapping):
                            if part.get("type") == "text" and isinstance(part.get("text"), str):
                                msg_content_chars += len(part["text"])
                            elif part.get("type") == "image_url":
                                image_tokens += 500
                                has_rich_data = True

                # Classify character counts
                if role in ("tool", "function"):
                    tool_chars += msg_content_chars
                    has_rich_data = True
                else:
                    messages_chars += msg_content_chars

                # Inline tool calls or function calls in messages
                for tc_key in ("tool_calls", "function_call"):
                    if tc_key in msg:
                        tc_val = msg[tc_key]
                        if tc_val is not None:
                            tool_chars += len(json.dumps(tc_val))
                            has_rich_data = True

                # Message level images (e.g. Ollama messages can have image arrays)
                if "images" in msg:
                    images_list = msg["images"]
                    if isinstance(images_list, list):
                        image_tokens += len(images_list) * 500
                        has_rich_data = True

        if messages_chars > 0:
            contributions["messages"] = math.ceil(messages_chars / chars_per_token)

        # 3. Tool and function definitions at root
        for tool_key in ("tools", "functions", "tool_choice"):
            if tool_key in body:
                tool_def = body[tool_key]
                if tool_def is not None:
                    tool_chars += len(json.dumps(tool_def))
                    has_rich_data = True

        if tool_chars > 0:
            contributions["tool_payloads"] = math.ceil(tool_chars / chars_per_token)

        # 4. Root level images (Ollama root images parameter)
        root_images = body.get("images")
        if isinstance(root_images, list):
            image_tokens += len(root_images) * 500
            has_rich_data = True

        if image_tokens > 0:
            contributions["other_payloads"] = image_tokens

        # Compute total
        total_tokens = sum(contributions.values())

        # Confidence heuristic
        # If there are tool definitions, tool calls, tool responses or images, confidence is lower.
        # Otherwise, if we have text-only prompts/messages, confidence is higher (0.8).
        if has_rich_data:
            confidence = 0.5
        elif total_tokens > 0:
            confidence = 0.8
        else:
            confidence = 1.0

        return TokenEstimationResult(
            estimated_tokens=total_tokens,
            confidence=confidence,
            field_contribution=contributions,
        )
