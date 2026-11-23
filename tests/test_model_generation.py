from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError

import openwebui_mcp.main as main
from openwebui_mcp.client import OpenWebUIClient
from openwebui_mcp.model_generation import (
    build_plain_completion_payload,
    extract_completion_text,
    run_saved_model_generation_test,
)


def test_saved_model_generation_schema_bounds_tokens_and_prompt() -> None:
    ok = main.SavedModelGenerationTestParam(model_id="m", prompt="hi")
    assert ok.max_output_tokens == 128

    with pytest.raises(ValidationError):
        main.SavedModelGenerationTestParam(model_id="m", prompt="")
    with pytest.raises(ValidationError):
        main.SavedModelGenerationTestParam(model_id="m", prompt="x", max_output_tokens=0)
    with pytest.raises(ValidationError):
        main.SavedModelGenerationTestParam(model_id="m", prompt="x", max_output_tokens=513)


def test_plain_completion_payload_has_no_persistence_or_tools() -> None:
    payload = build_plain_completion_payload(
        "custom-model",
        "Say OK",
        max_output_tokens=64,
    )

    assert payload == {
        "model": "custom-model",
        "messages": [{"role": "user", "content": "Say OK"}],
        "stream": False,
        "max_tokens": 64,
        "tools": [],
        "tool_choice": "none",
    }
    assert "chat_id" not in payload
    assert "metadata" not in payload
    assert "session_id" not in payload


def test_extract_completion_text_normalizes_openai_shape() -> None:
    assert extract_completion_text({"choices": [{"message": {"content": "Hello"}}]}) == "Hello"
    assert extract_completion_text({"choices": [{"message": {"content": "   "}}]}) is None
    assert extract_completion_text({"unexpected": True}) is None


@pytest.mark.asyncio
async def test_run_saved_model_generation_missing_model() -> None:
    client = AsyncMock()
    client._get_model_record_or_none = AsyncMock(return_value=None)

    result = await run_saved_model_generation_test(
        client,
        "missing",
        "hi",
        max_output_tokens=32,
        api_key="token",
    )

    assert result["generation"] == "failed"
    assert result["failure_reason"] == "model_not_found"
    assert result["version_pinning"]["available"] is False
    client.post.assert_not_called()


@pytest.mark.asyncio
async def test_run_saved_model_generation_passes_with_version_caveat() -> None:
    client = AsyncMock()
    client._get_model_record_or_none = AsyncMock(
        return_value={"id": "custom", "version_id": "ver-1"}
    )
    client.post = AsyncMock(return_value={"choices": [{"message": {"content": "pong"}}]})

    result = await run_saved_model_generation_test(
        client,
        "custom",
        "ping",
        max_output_tokens=16,
        api_key="token",
    )

    assert result["generation"] == "passed"
    assert result["output"] == "pong"
    assert result["model_version_observed_before_call"] == "ver-1"
    assert "no atomic version pin" in result["version_pinning"]["note"]
    client.post.assert_awaited_once()
    payload = client.post.await_args.kwargs["json"]
    assert payload["messages"] == [{"role": "user", "content": "ping"}]
    assert payload["tools"] == []


@pytest.mark.asyncio
async def test_run_saved_model_generation_malformed_response_fails_safely() -> None:
    client = AsyncMock()
    client._get_model_record_or_none = AsyncMock(return_value={"id": "custom"})
    client.post = AsyncMock(return_value={"not_openai": True})

    result = await run_saved_model_generation_test(
        client,
        "custom",
        "hi",
        max_output_tokens=32,
        api_key="token",
    )

    assert result["generation"] == "failed"
    assert result["failure_reason"] == "malformed_or_empty_response"
    assert "not_openai" not in str(result)


@pytest.mark.asyncio
async def test_run_saved_model_generation_upstream_error_is_generic() -> None:
    client = AsyncMock()
    client._get_model_record_or_none = AsyncMock(return_value={"id": "custom"})
    request = httpx.Request("POST", "https://webui.example/api/chat/completions")
    response = httpx.Response(502, request=request, text='{"detail":"secret leak"}')
    client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError("502", request=request, response=response)
    )

    result = await run_saved_model_generation_test(
        client,
        "custom",
        "hi",
        max_output_tokens=32,
        api_key="token",
    )

    assert result["generation"] == "failed"
    assert result["failure_reason"] == "upstream_error"
    assert "secret" not in str(result)


@pytest.mark.asyncio
async def test_client_test_saved_model_generation_wires_runner() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client._get_model_record_or_none = AsyncMock(return_value={"id": "custom", "version_id": "v1"})
    client.post = AsyncMock(return_value={"choices": [{"message": {"content": "ok"}}]})

    result = await client.test_saved_model_generation("custom", "hi", api_key="token")

    assert result["generation"] == "passed"
    assert client.post.await_args.args[0] == "/api/chat/completions"


def test_member_profile_excludes_generation_smoke_test() -> None:
    assert "test_saved_model_generation" not in main.MEMBER_PROFILE_TOOLS
