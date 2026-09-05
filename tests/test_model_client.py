from unittest.mock import AsyncMock, patch

import pytest

from openwebui_mcp.client import OpenWebUIClient


@pytest.mark.asyncio
async def test_get_model_uses_query_parameter_for_slash_safe_id() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(return_value={"id": "provider/model"})

    await client.get_model("provider/model", "token")

    client.get.assert_awaited_once_with("/api/v1/models/model?id=provider%2Fmodel", "token")


@pytest.mark.asyncio
async def test_list_models_normalizes_user_scoped_response_and_classifies_custom_models() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        side_effect=[
            {
                "data": [
                    {
                        "id": "custom-helper",
                        "name": "Custom Helper",
                        "owned_by": "openai",
                        "urlIdx": 2,
                        "is_active": False,
                        "ignored": "field",
                    },
                    {
                        "id": "ollama/llama3",
                        "name": "Llama 3",
                        "owned_by": "ollama",
                        "connection_id": "ollama-prod",
                    },
                ]
            },
            [{"id": "custom-helper"}],
        ]
    )

    result = await client.list_models(api_key="token")

    assert result == {
        "data": [
            {
                "id": "custom-helper",
                "name": "Custom Helper",
                "kind": "custom",
                "provider": "openai",
                "connection_id": 2,
                "is_active": False,
            },
            {
                "id": "ollama/llama3",
                "name": "Llama 3",
                "kind": "base",
                "provider": "ollama",
                "connection_id": "ollama-prod",
                "is_active": True,
            },
        ]
    }
    assert client.get.await_args_list[0].args == ("/api/models", "token")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"kind": "custom"}, ["custom-helper"]),
        ({"kind": "base"}, ["ollama/llama3"]),
        ({"provider": "ollama"}, ["ollama/llama3"]),
        ({"connection_id": "ollama-prod"}, ["ollama/llama3"]),
        ({"query": "LLAMA"}, ["ollama/llama3"]),
        ({"model_id": "helper"}, ["custom-helper"]),
        ({"display_name": "custom"}, ["custom-helper"]),
        ({"status": "inactive"}, ["custom-helper"]),
    ],
)
async def test_list_models_supports_all_filters(kwargs: dict, expected: list[str]) -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        side_effect=[
            {
                "items": [
                    {
                        "id": "custom-helper",
                        "name": "Custom Helper",
                        "provider": "openai",
                        "is_active": False,
                    },
                    {
                        "id": "ollama/llama3",
                        "name": "Llama 3",
                        "provider": "ollama",
                        "connection_id": "ollama-prod",
                    },
                ]
            },
            {"items": [{"id": "custom-helper"}]},
        ]
    )

    result = await client.list_models(api_key="token", **kwargs)

    assert [item["id"] for item in result["data"]] == expected


@pytest.mark.asyncio
async def test_update_model_access_uses_access_endpoint_for_provider_base_and_custom_ids() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.post = AsyncMock(return_value={"ok": True})

    for model_id in ("ollama/llama3", "gpt-4", "custom-helper"):
        await client.update_model_access(
            model_id,
            [{"type": "group", "id": "research"}],
            name="Model",
            api_key="token",
        )

    assert client.post.await_count == 3
    for call, model_id in zip(
        client.post.await_args_list, ("ollama/llama3", "gpt-4", "custom-helper")
    ):
        assert call.args == ("/api/v1/models/model/access/update", "token")
        assert call.kwargs["json"] == {
            "id": model_id,
            "name": "Model",
            "access_grants": [{"type": "group", "id": "research"}],
        }


@pytest.mark.asyncio
async def test_update_model_preserves_full_form_and_changes_base_model() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get_model = AsyncMock(
        return_value={
            "id": "research-assistant",
            "name": "Research Assistant",
            "base_model_id": "gpt-5.6-luna",
            "meta": {"description": "Existing description"},
            "params": {"temperature": 0.4, "system": "Existing system"},
            "access_grants": [{"type": "group", "id": "research"}],
            "is_active": True,
        }
    )
    client.post = AsyncMock(return_value={"id": "research-assistant"})

    await client.update_model(
        "research-assistant",
        base_model_id="gpt-5.6-terra",
        params={"temperature": 0.7},
        api_key="token",
    )

    client.post.assert_awaited_once_with(
        "/api/v1/models/model/update",
        "token",
        json={
            "id": "research-assistant",
            "name": "Research Assistant",
            "base_model_id": "gpt-5.6-terra",
            "meta": {"description": "Existing description"},
            "params": {"temperature": 0.7, "system": "Existing system"},
            "access_grants": [{"type": "group", "id": "research"}],
            "is_active": True,
        },
    )


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

    with patch("openwebui_mcp.client.httpx.AsyncClient", return_value=AsyncClient()):
        result = await client.get("/api/v1/models/export")

    assert result == {"data": [{"id": "model-a"}]}


@pytest.mark.asyncio
async def test_upload_text_file_forwards_session_token_and_multipart_payload() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")

    class Response:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"id": "file-1"}

    class AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            return Response()

    fake = AsyncClient()
    with patch("openwebui_mcp.client.httpx.AsyncClient", return_value=fake):
        result = await client.upload_text_file("update.md", "# Update", "session-token")

    assert result == {"id": "file-1"}
    assert fake.args == ("https://webui.example/api/v1/files/",)
    assert fake.kwargs["headers"] == {"Authorization": "Bearer session-token"}
    assert fake.kwargs["data"] == {"process": "true", "process_in_background": "false"}
    assert fake.kwargs["files"]["file"][0] == "update.md"
    assert fake.kwargs["files"]["file"][1] == b"# Update"


@pytest.mark.asyncio
async def test_add_user_to_group_uses_user_ids_payload() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.post = AsyncMock(return_value={"id": "group-1"})

    await client.add_user_to_group("group-1", "user-1", "token")

    client.post.assert_awaited_once_with(
        "/api/v1/groups/id/group-1/users/add",
        "token",
        json={"user_ids": ["user-1"]},
    )
