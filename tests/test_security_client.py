from unittest.mock import AsyncMock, patch

import httpx
import pytest

from openwebui_mcp.client import OpenWebUIClient
from openwebui_mcp.security import (
    project_config_export,
    project_file_search_record,
    project_models_config,
    sanitize_connection_path,
    sanitize_current_user,
    sanitize_error_detail,
    sanitize_service_url,
    sanitize_value,
)


def test_project_file_search_record_extracts_allowlisted_fields() -> None:
    record = project_file_search_record(
        {
            "id": "file-1",
            "filename": "notes.pdf",
            "user_id": "user-1",
            "meta": {
                "content_type": "application/pdf",
                "size": 1200,
                "data": {"knowledge_id": "kb-1"},
                "secret": "omit",
            },
            "data": {"status": "completed", "content": "secret"},
        }
    )

    assert record == {
        "id": "file-1",
        "filename": "notes.pdf",
        "knowledge_ids": ["kb-1"],
        "status": "completed",
        "content_type": "application/pdf",
        "size": 1200,
    }
    assert "meta" not in record
    assert "user_id" not in record


def test_sanitize_current_user_omits_session_token_entirely() -> None:
    payload = {
        "id": "user-1",
        "email": "user@example.com",
        "name": "User",
        "role": "admin",
        "token": "jwt-session-token-value",
        "permissions": {"chat": True},
        "unexpected": "field",
    }

    result = sanitize_current_user(payload)

    assert result["id"] == "user-1"
    assert "token" not in result
    assert "jwt-session-token-value" not in str(result)
    assert "unexpected" not in result


def test_sanitize_value_bounds_nested_response_size_and_depth() -> None:
    nested: dict[str, object] = {"apiKey": "must-not-escape"}
    for _ in range(25):
        nested = {"nested": nested}

    result = sanitize_value(
        {
            "deep": nested,
            "long": "x" * 3000,
            "items": list(range(500)),
            "mapping": {f"item-{index}": index for index in range(500)},
        }
    )

    assert result["long"] == "x" * 1999 + "…"
    assert len(result["items"]) == 201
    assert result["items"][-1] == "[truncated]"
    assert len(result["mapping"]) == 201
    assert result["mapping"]["_truncated"] is True
    assert "must-not-escape" not in str(result)
    assert "[truncated]" in str(result["deep"])


def test_nested_allowlisted_config_redacts_generic_credential_key_shapes() -> None:
    config = {
        "models.default_metadata": {
            "serviceKey": "svc-secret",
            "accessKeyId": "AKIA123",
            "AWS_ACCESS_KEY_ID": "also-hidden",
            "tokens": ["jwt-a"],
            "credentials": {"user": "x", "pass": "y"},
            "service_key": "another",
            "title": "Safe",
        },
    }

    result = project_config_export(config)
    meta = result["models.default_metadata"]

    assert meta["serviceKey"] == {"present": True}
    assert meta["accessKeyId"] == {"present": True}
    assert meta["AWS_ACCESS_KEY_ID"] == {"present": True}
    assert meta["tokens"] == {"present": True}
    assert meta["credentials"] == {"present": True}
    assert meta["service_key"] == {"present": True}
    assert meta["title"] == "Safe"
    assert "svc-secret" not in str(result)
    assert "AKIA123" not in str(result)


def test_nested_allowlisted_config_redacts_camelcase_credential_keys() -> None:
    config = {
        "models.default_metadata": {
            "apiKey": "super-secret",
            "clientSecret": "another-secret",
            "accessToken": "token-value",
            "privateKey": "pem-body",
            "title": "My model",
        },
    }

    result = project_config_export(config)

    meta = result["models.default_metadata"]
    assert meta["apiKey"] == {"present": True}
    assert meta["clientSecret"] == {"present": True}
    assert meta["accessToken"] == {"present": True}
    assert meta["privateKey"] == {"present": True}
    assert meta["title"] == "My model"
    assert "super-secret" not in str(result)


def test_project_config_export_uses_allowlist_and_presence_only() -> None:
    config = {
        "openai.api_base_urls": ["https://api.example/v1"],
        "openai.api_keys": ["super-secret-key"],
        "ui.banners": [],
        "ui.prompt_suggestions": [{"content": "Hi"}],
        "rag.tavily_api_key": "also-hidden",
    }

    result = project_config_export(config)

    assert result == {
        "ui.banners": [],
        "ui.prompt_suggestions": [{"content": "Hi"}],
        "openai.api_keys": {"present": True},
        "openai.api_base_urls": ["https://api.example/v1"],
    }
    assert "super-secret-key" not in str(result)
    assert "also-hidden" not in str(result)


def test_sanitize_service_url_strips_x_api_key_query() -> None:
    url = sanitize_service_url("https://api.example/v1?x-api-key=secret&model=gpt")
    assert url is not None
    assert "x-api-key" not in url
    assert "secret" not in url
    assert "model=gpt" in url


def test_sanitize_service_url_strips_userinfo_and_query_secrets() -> None:
    url = sanitize_service_url("https://user:pass@api.example/v1?api_key=secret&model=gpt")
    assert "user:pass" not in (url or "")
    assert "api_key" not in (url or "")
    assert "api.example" in (url or "")
    assert "model=gpt" in (url or "")


def test_sanitize_error_detail_omits_non_json_bodies() -> None:
    assert sanitize_error_detail("upstream html error page") == "Request failed."


def test_sanitize_error_detail_never_echoes_json_message_fields() -> None:
    sanitized = sanitize_error_detail(
        '{"detail":"Bearer sekret-token-value api_key=abc","message":"leaked"}',
        request_token="sekret-token-value",
    )

    assert sanitized == "Request failed."
    assert "sekret" not in sanitized
    assert "api_key" not in sanitized
    assert "leaked" not in sanitized


def test_sanitize_service_url_redacts_schemeless_sensitive_urls() -> None:
    assert sanitize_service_url("api.example/v1?api_key=secret") == "[redacted-url]"


@pytest.mark.asyncio
async def test_search_files_requests_metadata_only_and_strips_content() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        return_value={
            "data": [
                {
                    "id": "file-1",
                    "filename": "notes.md",
                    "data": {"content": "extracted"},
                }
            ]
        }
    )

    result = await client.search_files("*.md", "token")

    client.get.assert_awaited_once_with(
        "/api/v1/files/search",
        "token",
        params={"filename": "*.md", "content": False, "skip": 0, "limit": 100},
    )
    hit = result["data"][0]
    assert hit == {"id": "file-1", "filename": "notes.md"}
    assert "data" not in hit
    assert "meta" not in hit


@pytest.mark.asyncio
async def test_get_file_content_enforces_knowledge_scope() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.list_knowledge_files = AsyncMock(
        return_value={
            "items": [{"id": "other-file"}],
            "total": 1,
            "page": 1,
            "limit": 30,
            "next_page": None,
        }
    )

    with pytest.raises(ValueError, match="not attached"):
        await client.get_file_content("file-1", "token", knowledge_id="kb-1")

    client.list_knowledge_files.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_file_content_allows_linked_knowledge_file() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.list_knowledge_files = AsyncMock(
        return_value={
            "items": [{"id": "file-1"}],
            "total": 1,
            "page": 1,
            "limit": 30,
            "next_page": None,
        }
    )
    client.get = AsyncMock(return_value={"content": "hello"})

    result = await client.get_file_content("file-1", "token", knowledge_id="kb-1")

    assert result["content"] == "hello"
    assert result["total_lines"] == 1
    assert result["line_count"] == 1
    client.list_knowledge_files.assert_awaited_once()
    assert client.get.await_args_list[0].args == ("/api/v1/files/file-1/data/content", "token")


@pytest.mark.asyncio
async def test_search_files_honors_skip_and_limit() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(return_value={"data": []})

    await client.search_files("*.pdf", "token", skip=10, limit=50)

    client.get.assert_awaited_once_with(
        "/api/v1/files/search",
        "token",
        params={"filename": "*.pdf", "content": False, "skip": 10, "limit": 50},
    )


@pytest.mark.asyncio
async def test_get_config_redacts_sensitive_export_values() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        return_value={
            "openai.api_keys": ["hidden"],
            "rag.tavily_api_key": "also-hidden",
            "ui.banners": [],
        }
    )

    result = await client.get_config("token")

    assert "hidden" not in str(result)
    assert "also-hidden" not in str(result)
    assert result == {"ui.banners": [], "openai.api_keys": {"present": True}}
    assert "rag.tavily_api_key" not in result


@pytest.mark.asyncio
async def test_get_tool_servers_redacts_auth_material() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        return_value={
            "TOOL_SERVER_CONNECTIONS": [
                {
                    "url": "http://127.0.0.1:8000/mcp",
                    "type": "mcp",
                    "key": "mcp-secret",
                    "headers": {"Authorization": "Bearer abc"},
                    "info": {"id": "local", "description": "Local"},
                    "config": {"enable": True, "function_name_filter_list": ["ping"]},
                }
            ]
        }
    )

    result = await client.get_tool_servers("token")

    assert result["TOOL_SERVER_CONNECTIONS"][0]["info"]["id"] == "local"
    assert result["TOOL_SERVER_CONNECTIONS"][0]["key"] == {"present": True}
    assert "mcp-secret" not in str(result)
    assert "abc" not in str(result)


def test_sanitize_connection_path_strips_query_and_userinfo() -> None:
    assert sanitize_connection_path("/mcp?api_key=secret") == "/mcp"
    assert sanitize_connection_path("user:pass@/mcp") == "/mcp"


@pytest.mark.asyncio
async def test_update_model_empty_builtin_tools_dict_is_noop() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get_model = AsyncMock(
        return_value={
            "id": "custom",
            "name": "Custom",
            "base_model_id": "gpt-5",
            "meta": {"builtinTools": {"memory": True, "web_search": True}},
            "params": {},
            "access_grants": [],
            "is_active": True,
        }
    )
    client.post = AsyncMock(return_value={"id": "custom"})

    await client.update_model("custom", builtin_tools={}, api_key="token")

    payload = client.post.await_args.kwargs["json"]
    assert payload["meta"]["builtinTools"] == {"memory": True, "web_search": True}


@pytest.mark.asyncio
async def test_update_model_merges_builtin_tools_and_preserves_sibling_switches() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get_model = AsyncMock(
        return_value={
            "id": "custom",
            "name": "Custom",
            "base_model_id": "gpt-5",
            "meta": {"builtinTools": {"memory": True, "web_search": True}},
            "params": {},
            "access_grants": [],
            "is_active": True,
        }
    )
    client.post = AsyncMock(return_value={"id": "custom"})

    await client.update_model("custom", builtin_tools={"memory": False}, api_key="token")

    payload = client.post.await_args.kwargs["json"]
    assert payload["meta"]["builtinTools"] == {"memory": False, "web_search": True}


@pytest.mark.asyncio
async def test_update_model_merges_capabilities_and_clears_skill_ids() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get_model = AsyncMock(
        return_value={
            "id": "custom",
            "name": "Custom",
            "base_model_id": "gpt-5",
            "meta": {
                "capabilities": {"vision": True, "web_search": True},
                "skillIds": ["skill-1"],
            },
            "params": {},
            "access_grants": [],
            "is_active": True,
        }
    )
    client.get = AsyncMock()
    client.post = AsyncMock(return_value={"id": "custom"})

    await client.update_model(
        "custom",
        capabilities={"vision": False},
        skill_ids=[],
        api_key="token",
    )

    client.get.assert_not_called()
    payload = client.post.await_args.kwargs["json"]
    assert payload["meta"]["capabilities"] == {"vision": False, "web_search": True}
    assert "skillIds" not in payload["meta"]


@pytest.mark.asyncio
async def test_request_error_never_echoes_request_token() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    token = "super-secret-token"
    request = httpx.Request("POST", "https://webui.example/api/v1/configs/import")
    response = httpx.Response(
        400,
        request=request,
        headers={"content-type": "application/json"},
        content=b'{"openai.api_keys":["leaked"],"detail":"Bearer super-secret-token"}',
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
            await client.post("/api/v1/configs/import", token, json={})

    assert token not in str(error.value)
    assert "leaked" not in str(error.value)
    assert "super-secret-token" not in str(error.value)


def test_project_models_config_allowlists_fields() -> None:
    payload = {
        "DEFAULT_MODELS": "gpt-4",
        "MODEL_ORDER_LIST": ["a", "b"],
        "SECRET_SIDE_CHANNEL": "nope",
    }

    assert project_models_config(payload) == {
        "DEFAULT_MODELS": "gpt-4",
        "MODEL_ORDER_LIST": ["a", "b"],
    }
