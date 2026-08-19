from unittest.mock import AsyncMock

import pytest

from openwebui_mcp.client import OpenWebUIClient


@pytest.mark.asyncio
async def test_get_model_uses_query_parameter_for_slash_safe_id() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(return_value={"id": "provider/model"})

    await client.get_model("provider/model", "token")

    client.get.assert_awaited_once_with("/api/v1/models/model?id=provider%2Fmodel", "token")


@pytest.mark.asyncio
async def test_request_wraps_json_lists_for_mcp_structured_content() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")

    class Response:
        headers = {"content-type": "application/json"}

        def raise_for_status(self) -> None:
            pass

        def json(self) -> list[dict[str, str]]:
            return [{"id": "model-a"}]

    class AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def request(self, *args, **kwargs):
            return Response()

    from unittest.mock import patch

    with patch("openwebui_mcp.client.httpx.AsyncClient", return_value=AsyncClient()):
        result = await client.get("/api/v1/models/export")

    assert result == {"data": [{"id": "model-a"}]}


@pytest.mark.asyncio
async def test_update_model_preserves_required_fields_and_merges_parameters() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get_model = AsyncMock(
        return_value={
            "id": "writing-coach",
            "name": "Writing Coach",
            "base_model_id": "gpt-5",
            "meta": {"description": "Existing description"},
            "params": {"temperature": 0.4},
            "access_grants": [{"type": "group", "id": "editors"}],
            "is_active": True,
        }
    )
    client.post = AsyncMock(return_value={"id": "writing-coach"})

    await client.update_model(
        "writing-coach",
        params={"system": "Help with writing", "temperature": 0.7},
        api_key="token",
    )

    client.post.assert_awaited_once_with(
        "/api/v1/models/model/update",
        "token",
        json={
            "id": "writing-coach",
            "name": "Writing Coach",
            "base_model_id": "gpt-5",
            "meta": {"description": "Existing description"},
            "params": {"temperature": 0.7, "system": "Help with writing"},
            "access_grants": [{"type": "group", "id": "editors"}],
            "is_active": True,
        },
    )


@pytest.mark.asyncio
async def test_update_prompt_uses_id_and_preserves_required_fields() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get_prompt = AsyncMock(
        return_value={
            "id": "prompt-123",
            "command": "/coach",
            "name": "Writing coach",
            "content": "Existing content",
            "data": {"audience": "internal"},
            "meta": {"description": "Existing prompt"},
            "tags": ["writing"],
            "access_grants": [],
        }
    )
    client.post = AsyncMock(return_value={"id": "prompt-123"})

    await client.update_prompt("prompt-123", content="Updated content", api_key="token")

    client.post.assert_awaited_once_with(
        "/api/v1/prompts/id/prompt-123/update",
        "token",
        json={
            "command": "/coach",
            "name": "Writing coach",
            "content": "Updated content",
            "data": {"audience": "internal"},
            "meta": {"description": "Existing prompt"},
            "tags": ["writing"],
            "access_grants": [],
        },
    )
