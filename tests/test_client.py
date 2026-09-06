from unittest.mock import AsyncMock, patch

import httpx
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
async def test_search_files_passes_filename_as_encoded_query_parameter() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(return_value={"data": []})

    await client.search_files("quarterly report & notes/*.pdf", "token")

    client.get.assert_awaited_once_with(
        "/api/v1/files/search",
        "token",
        params={"filename": "quarterly report & notes/*.pdf"},
    )


@pytest.mark.asyncio
async def test_file_and_knowledge_ids_are_encoded_as_path_segments() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(return_value={})
    client.delete = AsyncMock(return_value={})

    await client.get_file("folder/file id")
    await client.delete_knowledge("tenant/knowledge id")

    assert client.get.await_args.args == ("/api/v1/files/folder%2Ffile%20id", None)
    client.delete.assert_awaited_once_with(
        "/api/v1/knowledge/tenant%2Fknowledge%20id/delete", None
    )


@pytest.mark.asyncio
async def test_mutation_routes_match_current_open_webui_source() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.delete = AsyncMock(return_value={})
    client.post = AsyncMock(return_value={})

    await client.delete_group("group-1", "token")
    await client.delete_tool("tool-1", "token")
    await client.delete_function("function-1", "token")
    await client.create_folder("Research", api_key="token")
    await client.archive_chat("chat-1", "token")
    await client.clone_chat("chat-1", "token")

    assert [call.args for call in client.delete.await_args_list] == [
        ("/api/v1/groups/id/group-1/delete", "token"),
        ("/api/v1/tools/id/tool-1/delete", "token"),
        ("/api/v1/functions/id/function-1/delete", "token"),
    ]
    assert [call.args for call in client.post.await_args_list] == [
        ("/api/v1/folders/", "token"),
        ("/api/v1/chats/chat-1/archive", "token"),
        ("/api/v1/chats/chat-1/clone", "token"),
    ]


@pytest.mark.asyncio
async def test_create_folder_maps_project_configuration() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.post = AsyncMock(return_value={"id": "folder-1"})

    await client.create_folder(
        "Research",
        system_prompt="Use primary sources.",
        knowledge_ids=["kb-1"],
        api_key="token",
    )

    client.post.assert_awaited_once_with(
        "/api/v1/folders/",
        "token",
        json={
            "name": "Research",
            "data": {
                "system_prompt": "Use primary sources.",
                "files": [{"id": "kb-1", "type": "collection"}],
            },
        },
    )


@pytest.mark.asyncio
async def test_update_folder_preserves_non_collection_attachments() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get_folder = AsyncMock(
        return_value={
            "data": {
                "system_prompt": "Old prompt",
                "files": [
                    {"id": "old-kb", "type": "collection"},
                    {"id": "file-1", "type": "file"},
                    {"id": "note-1", "type": "note"},
                ],
                "other": "preserved",
            }
        }
    )
    client.post = AsyncMock(return_value={"id": "folder-1"})

    await client.update_folder(
        "folder-1",
        system_prompt="New prompt",
        knowledge_ids=["new-kb"],
        api_key="token",
    )

    client.post.assert_awaited_once_with(
        "/api/v1/folders/folder-1/update",
        "token",
        json={
            "data": {
                "system_prompt": "New prompt",
                "files": [
                    {"id": "file-1", "type": "file"},
                    {"id": "note-1", "type": "note"},
                    {"id": "new-kb", "type": "collection"},
                ],
                "other": "preserved",
            }
        },
    )


@pytest.mark.asyncio
async def test_create_tool_always_sends_metadata_object() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.post = AsyncMock(return_value={"id": "weather_tool"})

    await client.create_tool("weather_tool", "Weather", "class Tools: pass", api_key="token")

    client.post.assert_awaited_once_with(
        "/api/v1/tools/create",
        "token",
        json={"id": "weather_tool", "name": "Weather", "content": "class Tools: pass", "meta": {}},
    )


@pytest.mark.asyncio
async def test_update_tool_sends_complete_form_and_preserves_metadata() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get_tool = AsyncMock(
        return_value={
            "id": "weather_tool",
            "name": "Existing Weather",
            "content": "class Tools: pass",
            "meta": {"description": "Existing description", "manifest": {}},
        }
    )
    client.post = AsyncMock(return_value={"id": "weather_tool"})

    await client.update_tool("weather_tool", name="Updated Weather", api_key="token")

    client.get_tool.assert_awaited_once_with("weather_tool", "token")
    client.post.assert_awaited_once_with(
        "/api/v1/tools/id/weather_tool/update",
        "token",
        json={
            "id": "weather_tool",
            "name": "Updated Weather",
            "content": "class Tools: pass",
            "meta": {"description": "Existing description", "manifest": {}},
        },
    )


@pytest.mark.asyncio
async def test_request_error_preserves_body_without_credentials() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    token = "super-secret-token"
    request = httpx.Request("POST", "https://webui.example/api/v1/tools/create")
    response = httpx.Response(
        422,
        request=request,
        headers={"content-type": "application/json"},
        content=b'{"detail":"meta is required"}',
    )

    class AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def request(self, *args, **kwargs):
            return response

    with patch("openwebui_mcp.client.httpx.AsyncClient", return_value=AsyncClient()):
        with pytest.raises(httpx.HTTPStatusError) as error:
            await client.post("/api/v1/tools/create", token, json={})

    assert 'meta is required' in str(error.value)
    assert token not in str(error.value)


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
async def test_update_tool_server_config_preserves_other_connections_and_fields() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        return_value={
            "TOOL_SERVER_CONNECTIONS": [
                {
                    "url": "http://member:8000/mcp",
                    "type": "mcp",
                    "info": {"id": "member", "description": "Old"},
                    "config": {"enable": True, "function_name_filter_list": ["old"]},
                },
                {"url": "http://other:8000/mcp", "info": {"id": "other"}},
            ]
        }
    )
    client.post = AsyncMock(return_value={"ok": True})

    await client.update_tool_server_config(
        "member",
        ["list_folders", "update_folder"],
        description="Permission-scoped model, knowledge, and project management",
        api_key="token",
    )

    client.post.assert_awaited_once_with(
        "/api/v1/configs/tool_servers",
        "token",
        json={
            "TOOL_SERVER_CONNECTIONS": [
                {
                    "url": "http://member:8000/mcp",
                    "type": "mcp",
                    "info": {
                        "id": "member",
                        "description": "Permission-scoped model, knowledge, and project management",
                    },
                    "config": {
                        "enable": True,
                        "function_name_filter_list": ["list_folders", "update_folder"],
                    },
                },
                {"url": "http://other:8000/mcp", "info": {"id": "other"}},
            ]
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
