"""Plain saved custom-model generation smoke test (no chat persistence)."""

from __future__ import annotations

from typing import Any, Literal

import httpx

GenerationStatus = Literal["passed", "failed"]

_VERSION_PIN_NOTE = (
    "Open WebUI exposes no atomic version pin for POST /api/chat/completions; "
    "the model record may change before the provider call completes."
)


def observed_model_version(model: dict[str, Any]) -> Any:
    """Best-effort version marker from the model record (may be absent)."""
    for key in ("version_id", "versionId", "updated_at", "updatedAt"):
        value = model.get(key)
        if value is not None:
            return value
    return None


def build_plain_completion_payload(
    model_id: str,
    prompt: str,
    *,
    max_output_tokens: int,
) -> dict[str, Any]:
    """Minimal OpenAI-style body for a non-streaming, tool-free completion."""
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_output_tokens,
        "tools": [],
        "tool_choice": "none",
    }


def extract_completion_text(response: Any) -> str | None:
    """Return assistant text from a chat completion JSON body, or None if unusable."""
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return None


def _base_result(model_id: str, *, model_version: Any = None) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "generation": "failed",
        "output": None,
        "native_tool_execution": "not_tested",
        "browser_rendering": "not_tested",
        "model_version_observed_before_call": model_version,
        "version_pinning": {
            "available": False,
            "note": _VERSION_PIN_NOTE,
        },
    }


def model_not_found_result(model_id: str) -> dict[str, Any]:
    result = _base_result(model_id)
    result["failure_reason"] = "model_not_found"
    return result


def generation_failed_result(
    model_id: str,
    *,
    model_version: Any,
    failure_reason: str,
) -> dict[str, Any]:
    result = _base_result(model_id, model_version=model_version)
    result["failure_reason"] = failure_reason
    return result


def generation_passed_result(
    model_id: str,
    *,
    model_version: Any,
    output: str,
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "generation": "passed",
        "output": output,
        "native_tool_execution": "not_tested",
        "browser_rendering": "not_tested",
        "model_version_observed_before_call": model_version,
        "version_pinning": {
            "available": False,
            "note": _VERSION_PIN_NOTE,
        },
    }


async def run_saved_model_generation_test(
    client: Any,
    model_id: str,
    prompt: str,
    *,
    max_output_tokens: int,
    api_key: str | None,
) -> dict[str, Any]:
    """Execute a bounded plain completion against a saved custom model."""
    model = await client._get_model_record_or_none(model_id, api_key)
    if not model:
        return model_not_found_result(model_id)

    model_version = observed_model_version(model)
    payload = build_plain_completion_payload(
        model_id,
        prompt,
        max_output_tokens=max_output_tokens,
    )

    try:
        response = await client.post("/api/chat/completions", api_key, json=payload)
    except httpx.HTTPStatusError:
        return generation_failed_result(
            model_id,
            model_version=model_version,
            failure_reason="upstream_error",
        )

    output = extract_completion_text(response)
    if output is None:
        return generation_failed_result(
            model_id,
            model_version=model_version,
            failure_reason="malformed_or_empty_response",
        )

    return generation_passed_result(
        model_id,
        model_version=model_version,
        output=output,
    )
