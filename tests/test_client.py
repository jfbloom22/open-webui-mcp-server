from unittest.mock import AsyncMock

import pytest

from openwebui_mcp.client import OpenWebUIClient


@pytest.mark.asyncio
async def test_list_users_fetches_all_admin_pages() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        side_effect=[
            {"users": [{"id": f"user-{index}"} for index in range(30)], "total": 45},
            {"users": [{"id": f"user-{index}"} for index in range(30, 45)], "total": 45},
        ]
    )

    result = await client.list_users(api_key="token")

    assert [user["id"] for user in result["users"]] == [f"user-{index}" for index in range(45)]
    assert result["total"] == 45
    assert client.get.await_args_list[0].args == ("/api/v1/users/?page=1", "token")
    assert client.get.await_args_list[1].args == ("/api/v1/users/?page=2", "token")


@pytest.mark.asyncio
async def test_list_files_fetches_all_pages_without_content() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        side_effect=[
            {
                "items": [
                    {"id": f"file-{index}", "data": {"content": "extracted text"}}
                    for index in range(50)
                ],
                "total": 55,
            },
            {"items": [{"id": f"file-{index}"} for index in range(50, 55)], "total": 55},
        ]
    )

    result = await client.list_files(api_key="token")

    assert [file["id"] for file in result["items"]] == [f"file-{index}" for index in range(55)]
    assert all("data" not in file for file in result["items"])
    assert result["total"] == 55
    assert client.get.await_args_list[0].args == ("/api/v1/files/?page=1&content=false", "token")
    assert client.get.await_args_list[1].args == ("/api/v1/files/?page=2&content=false", "token")


@pytest.mark.asyncio
async def test_list_knowledge_fetches_all_30_item_pages_and_compacts_records() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        side_effect=[
            {
                "items": [
                    {
                        "id": f"knowledge-{index}",
                        "name": f"Knowledge {index}",
                        "description": "description",
                        "access_grants": [{"id": "group-1"}],
                        "internal": "omitted",
                    }
                    for index in range(30)
                ],
                "total": 45,
            },
            {
                "items": [
                    {
                        "id": f"knowledge-{index}",
                        "name": f"Knowledge {index}",
                        "description": "description",
                        "access_grants": [],
                        "internal": "omitted",
                    }
                    for index in range(30, 45)
                ],
                "total": 45,
            },
        ]
    )

    result = await client.list_knowledge(api_key="token")

    assert [item["id"] for item in result["items"]] == [
        f"knowledge-{index}" for index in range(45)
    ]
    assert result["total"] == 45
    assert result["items"][0] == {
        "id": "knowledge-0",
        "name": "Knowledge 0",
        "description": "description",
        "access_grant_count": 1,
    }
    assert "internal" not in result["items"][0]
    assert [call.args for call in client.get.await_args_list] == [
        ("/api/v1/knowledge/?page=1", "token"),
        ("/api/v1/knowledge/?page=2", "token"),
    ]


@pytest.mark.asyncio
async def test_list_knowledge_stops_on_empty_page_and_handles_invalid_total() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        side_effect=[
            {"items": [{"id": "knowledge-1", "access_grants": None}], "total": 60},
            {"items": []},
        ]
    )

    result = await client.list_knowledge(api_key="token")

    assert result["items"] == [{"id": "knowledge-1"}]
    assert [call.args for call in client.get.await_args_list] == [
        ("/api/v1/knowledge/?page=1", "token"),
        ("/api/v1/knowledge/?page=2", "token"),
    ]

    client.get = AsyncMock(return_value={"items": [{"id": "knowledge-1"}], "total": "60"})
    result = await client.list_knowledge(api_key="token")

    assert result["items"] == [{"id": "knowledge-1", "access_grant_count": 0}]
    client.get.assert_awaited_once_with("/api/v1/knowledge/?page=1", "token")


@pytest.mark.asyncio
async def test_get_model_uses_query_parameter_for_slash_safe_id() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(return_value={"id": "provider/model"})

    await client.get_model("provider/model", "token")

    client.get.assert_awaited_once_with("/api/v1/models/model?id=provider%2Fmodel", "token")


@pytest.mark.asyncio
async def test_get_config_uses_current_open_webui_export_endpoint() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(return_value={"ui.banners": []})

    result = await client.get_config(api_key="token")

    assert result == {"ui.banners": []}
    client.get.assert_awaited_once_with("/api/v1/configs/export", "token")


@pytest.mark.asyncio
async def test_update_knowledge_access_uses_open_webui_access_form() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.post = AsyncMock(return_value={"id": "knowledge-1", "name": "Research"})
    grants = [{"principal_type": "group", "principal_id": "research", "permission": "write"}]

    result = await client.update_knowledge_access("knowledge-1", grants, "token")

    assert result == {"id": "knowledge-1", "name": "Research"}
    client.post.assert_awaited_once_with(
        "/api/v1/knowledge/knowledge-1/access/update",
        "token",
        json={"access_grants": grants},
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
