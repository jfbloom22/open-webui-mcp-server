from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from openwebui_mcp import main


def test_member_profile_allowlist_excludes_admin_surfaces() -> None:
    assert "list_knowledge_files" in main.MEMBER_PROFILE_TOOLS
    assert "get_file_processing_status" in main.MEMBER_PROFILE_TOOLS
    assert "get_environment_info" in main.MEMBER_PROFILE_TOOLS
    assert "validate_model_configuration" in main.MEMBER_PROFILE_TOOLS
    assert "update_model" in main.MEMBER_PROFILE_TOOLS
    assert "list_folders" in main.MEMBER_PROFILE_TOOLS
    assert "get_folder" in main.MEMBER_PROFILE_TOOLS
    assert "create_folder" in main.MEMBER_PROFILE_TOOLS
    assert "update_folder" in main.MEMBER_PROFILE_TOOLS
    assert "add_knowledge_file" in main.MEMBER_PROFILE_TOOLS
    assert "add_knowledge_text" not in main.MEMBER_PROFILE_TOOLS
    assert "update_knowledge_access" in main.MEMBER_PROFILE_TOOLS
    assert "list_users" not in main.MEMBER_PROFILE_TOOLS
    assert "get_tool_servers" not in main.MEMBER_PROFILE_TOOLS
    assert "update_model_access" not in main.MEMBER_PROFILE_TOOLS
    assert "get_model_connections" not in main.MEMBER_PROFILE_TOOLS
    assert "set_connection_model_ids" not in main.MEMBER_PROFILE_TOOLS
    assert "set_model_visibility" not in main.MEMBER_PROFILE_TOOLS
    assert "set_model_enabled" not in main.MEMBER_PROFILE_TOOLS
    assert not any(name.startswith("delete_") for name in main.MEMBER_PROFILE_TOOLS)
    assert "set_skill_active" not in main.MEMBER_PROFILE_TOOLS


def test_delete_skill_is_the_only_delete_tool_exempt_from_global_filter() -> None:
    registered = {"delete_skill", "delete_model", "set_skill_active"}
    globally_blocked = {
        name for name in registered if name.startswith("delete_") and name != "delete_skill"
    }

    assert globally_blocked == {"delete_model"}


def test_deployed_profiles_require_forwarded_session_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_PROFILE", "member")
    token = main._current_user_token.set(None)
    try:
        with pytest.raises(RuntimeError, match="forwarded Open WebUI session token"):
            main.get_user_token()
    finally:
        main._current_user_token.reset(token)


@pytest.mark.asyncio
async def test_http_middleware_forwards_session_token_for_scoped_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_PROFILE", "admin")
    observed: list[str | None] = []

    async def app(scope, receive, send):
        observed.append(main.get_user_token())

    middleware = main.AuthMiddleware(app)
    await middleware(
        {"type": "http", "headers": [(b"authorization", b"Bearer session-token")]},
        Mock(),
        Mock(),
    )

    assert observed == ["session-token"]
    with pytest.raises(RuntimeError, match="forwarded Open WebUI session token"):
        main.get_user_token()


@pytest.mark.asyncio
async def test_http_health_endpoint_is_live_without_mcp_auth() -> None:
    messages = []

    async def app(scope, receive, send):
        raise AssertionError("health endpoint should be handled by middleware")

    async def send(message):
        messages.append(message)

    middleware = main.AuthMiddleware(app, mcp_token="secret")
    await middleware(
        {"type": "http", "path": "/healthz", "method": "GET", "headers": []},
        Mock(),
        send,
    )

    assert messages[0]["status"] == 200
    assert messages[1]["body"] == b"ok"


@pytest.mark.asyncio
async def test_advertised_mcp_tool_names_uses_server_list_tools() -> None:
    registered = [SimpleNamespace(name="list_models"), SimpleNamespace(name="get_model")]
    list_tools = AsyncMock(return_value=registered)
    with patch.object(main.mcp, "list_tools", list_tools):
        names = await main.advertised_mcp_tool_names()

    list_tools.assert_awaited_once_with(run_middleware=False)
    assert names == ["get_model", "list_models"]


@pytest.mark.asyncio
async def test_get_environment_info_uses_advertised_tool_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENWEBUI_API_KEY", "session-token")

    client = Mock()
    client.get_environment_info = AsyncMock(
        return_value={
            "mcp_operations": {"active": ["list_models"]},
        }
    )
    with (
        patch.object(main, "get_client", return_value=client),
        patch.object(main, "advertised_mcp_tool_names", AsyncMock(return_value=["list_models"])),
    ):
        result = await main.get_environment_info(Mock())

    client.get_environment_info.assert_awaited_once()
    assert client.get_environment_info.await_args.kwargs["active_operations"] == ["list_models"]
    assert result["mcp_operations"]["active"] == ["list_models"]


@pytest.mark.asyncio
async def test_configure_member_profile_removes_unapproved_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_PROFILE", "member")
    monkeypatch.delenv("OPENWEBUI_DISABLED_TOOLS", raising=False)
    with (
        patch.object(
            main,
            "advertised_mcp_tool_names",
            AsyncMock(return_value=["list_models", "update_model", "delete_model", "delete_skill"]),
        ),
        patch.object(main.mcp.local_provider, "remove_tool", remove_tool := Mock()),
    ):
        await main.configure_tools()

    assert [call.args[0] for call in remove_tool.call_args_list] == [
        "delete_model",
        "delete_skill",
    ]


@pytest.mark.asyncio
async def test_configure_tools_applies_explicit_disables_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_PROFILE", "local")
    monkeypatch.setenv("OPENWEBUI_DISABLED_TOOLS", "list_users, missing_tool")
    with (
        patch.object(
            main,
            "advertised_mcp_tool_names",
            AsyncMock(return_value=["list_models", "list_users", "delete_model", "delete_skill"]),
        ),
        patch.object(main.mcp.local_provider, "remove_tool", remove_tool := Mock()),
    ):
        await main.configure_tools()

    assert [call.args[0] for call in remove_tool.call_args_list] == ["delete_model", "list_users"]
    assert "delete_skill" not in [call.args[0] for call in remove_tool.call_args_list]


_CREATE_BASE = {"id": "m", "name": "Model", "base_model_id": "gpt-5"}


@pytest.mark.parametrize(
    ("param_cls", "null_field", "base_kwargs"),
    [
        ("ModelCreateParam", "capabilities", _CREATE_BASE),
        ("ModelCreateParam", "builtin_tools", _CREATE_BASE),
        ("ModelCreateParam", "skill_ids", _CREATE_BASE),
        ("ModelCreateParam", "default_feature_ids", _CREATE_BASE),
        ("ModelUpdateParam", "capabilities", {"model_id": "m"}),
        ("ModelUpdateParam", "builtin_tools", {"model_id": "m"}),
        ("ModelUpdateParam", "skill_ids", {"model_id": "m"}),
        ("ModelUpdateParam", "default_feature_ids", {"model_id": "m"}),
    ],
)
def test_model_schemas_reject_explicit_null_workspace_meta_fields(
    param_cls: str,
    null_field: str,
    base_kwargs: dict,
) -> None:
    model_cls = getattr(main, param_cls)
    with pytest.raises(ValidationError):
        model_cls(**base_kwargs, **{null_field: None})


def test_model_schemas_allow_empty_workspace_meta_patch_collections() -> None:
    created = main.ModelCreateParam(
        id="m",
        name="Model",
        base_model_id="gpt-5",
        capabilities={},
        builtin_tools={},
        skill_ids=[],
        default_feature_ids=[],
    )
    updated = main.ModelUpdateParam(
        model_id="m",
        capabilities={},
        builtin_tools={},
        skill_ids=[],
        default_feature_ids=[],
    )

    assert created.capabilities == {}
    assert created.skill_ids == []
    assert updated.builtin_tools == {}
    assert updated.default_feature_ids == []


def test_model_schemas_allow_null_description_for_clear() -> None:
    created = main.ModelCreateParam(
        id="m",
        name="Model",
        base_model_id="gpt-5",
        description=None,
    )
    updated = main.ModelUpdateParam(model_id="m", description=None)

    assert created.description is None
    assert updated.description is None


def test_model_schemas_restrict_function_calling_values() -> None:
    omitted = main.ModelCreateParam(id="m", name="Model", base_model_id="gpt-5")
    assert omitted.function_calling is None

    native = main.ModelCreateParam(
        id="m",
        name="Model",
        base_model_id="gpt-5",
        function_calling="native",
    )
    assert native.function_calling == "native"

    with pytest.raises(ValidationError):
        main.ModelCreateParam(
            id="m",
            name="Model",
            base_model_id="gpt-5",
            function_calling="compat",
        )


def test_model_tool_schemas_expose_extended_workspace_fields() -> None:
    create_schema = main.ModelCreateParam.model_json_schema()
    update_schema = main.ModelUpdateParam.model_json_schema()

    for schema in (create_schema, update_schema):
        assert {
            "description",
            "function_calling",
            "capabilities",
            "builtin_tools",
            "skill_ids",
            "default_feature_ids",
        } <= set(schema["properties"])


def test_file_content_read_schema_exposes_knowledge_scope() -> None:
    schema = main.FileContentReadParam.model_json_schema()
    assert schema["properties"]["file_id"]["type"] == "string"
    assert "knowledge_id" in schema["properties"]


def test_model_tool_schemas_expose_kind_filters_and_access_grants() -> None:
    list_schema = main.ModelListParam.model_json_schema()
    access_schema = main.ModelAccessParam.model_json_schema()
    update_schema = main.ModelUpdateParam.model_json_schema()
    create_schema = main.ModelCreateParam.model_json_schema()

    assert list_schema["properties"]["kind"]["default"] == "all"
    assert list_schema["properties"]["kind"]["enum"] == ["all", "custom", "base"]
    assert set(("provider", "connection_id", "query", "model_id", "display_name", "status")) <= set(
        list_schema["properties"]
    )
    assert "access_grants" in access_schema["properties"]
    assert "name" in access_schema["properties"]
    assert "access_grants" in create_schema["properties"]
    assert "access_grants" in update_schema["properties"]
    assert "suggestion_prompts" in create_schema["properties"]
    assert "suggestion_prompts" in update_schema["properties"]
    for schema in (create_schema, update_schema):
        assert {
            "reasoning_effort",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        } <= set(schema["properties"])


def test_tool_server_config_schema_exposes_scoped_fields() -> None:
    schema = main.ToolServerConfigUpdateParam.model_json_schema()

    assert set(schema["properties"]) == {
        "server_id",
        "function_name_filter_list",
        "description",
    }


def test_model_management_admin_tool_schemas_are_explicit_and_guarded() -> None:
    connection_schema = main.ConnectionModelIdsUpdateParam.model_json_schema()
    visibility_schema = main.ModelVisibilityParam.model_json_schema()
    enabled_schema = main.ModelEnabledParam.model_json_schema()

    assert set(connection_schema["properties"]) == {
        "connection_index",
        "expected_base_url",
        "expected_model_ids",
        "model_ids",
    }
    assert connection_schema["properties"]["connection_index"]["minimum"] == 0
    assert "visible" in visibility_schema["properties"]
    assert "enabled" in enabled_schema["properties"]


@pytest.mark.asyncio
async def test_model_handlers_forward_sampling_and_reasoning_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.create_model = AsyncMock(return_value={"id": "new-model"})
    client.update_model = AsyncMock(return_value={"id": "existing-model"})
    monkeypatch.setattr(main, "get_client", lambda: client)
    monkeypatch.setenv("MCP_PROFILE", "local")
    monkeypatch.setenv("OPENWEBUI_API_KEY", "session-token")

    await main.create_model(
        main.ModelCreateParam(
            id="new-model",
            name="New Model",
            base_model_id="gpt-5.6-terra",
            reasoning_effort="high",
            top_p=0.8,
            frequency_penalty=0.1,
            presence_penalty=0.2,
        ),
        Mock(),
    )
    await main.update_model(
        main.ModelUpdateParam(
            model_id="existing-model",
            reasoning_effort="low",
            top_p=0.9,
            frequency_penalty=-0.1,
            presence_penalty=0.0,
        ),
        Mock(),
    )

    assert client.create_model.await_args.kwargs["params"] == {
        "reasoning_effort": "high",
        "top_p": 0.8,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
    }
    assert client.update_model.await_args.kwargs["params"] == {
        "reasoning_effort": "low",
        "top_p": 0.9,
        "frequency_penalty": -0.1,
        "presence_penalty": 0.0,
    }


@pytest.mark.asyncio
async def test_update_model_forwards_function_calling_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.update_model = AsyncMock(return_value={"id": "existing-model"})
    monkeypatch.setattr(main, "get_client", lambda: client)
    monkeypatch.setenv("MCP_PROFILE", "local")
    monkeypatch.setenv("OPENWEBUI_API_KEY", "session-token")

    await main.update_model(
        main.ModelUpdateParam(model_id="existing-model", function_calling="native"),
        Mock(),
    )

    assert client.update_model.await_args.kwargs["params"] == {"function_calling": "native"}


@pytest.mark.asyncio
async def test_model_handlers_forward_workspace_meta_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.create_model = AsyncMock(return_value={"id": "new-model"})
    client.update_model = AsyncMock(return_value={"id": "existing-model"})
    monkeypatch.setattr(main, "get_client", lambda: client)
    monkeypatch.setenv("MCP_PROFILE", "local")
    monkeypatch.setenv("OPENWEBUI_API_KEY", "session-token")

    await main.create_model(
        main.ModelCreateParam(
            id="new-model",
            name="New Model",
            base_model_id="gpt-5",
            description="Helper",
            function_calling="native",
            capabilities={"vision": False},
            builtin_tools={"memory": False},
            skill_ids=["skill-1"],
            default_feature_ids=["web_search"],
        ),
        Mock(),
    )
    await main.update_model(
        main.ModelUpdateParam(
            model_id="existing-model",
            capabilities={"web_search": True},
            skill_ids=[],
        ),
        Mock(),
    )

    create_kwargs = client.create_model.await_args.kwargs
    assert create_kwargs["params"]["function_calling"] == "native"
    assert create_kwargs["description"] == "Helper"
    assert create_kwargs["capabilities"] == {"vision": False}
    assert create_kwargs["builtin_tools"] == {"memory": False}
    assert create_kwargs["skill_ids"] == ["skill-1"]
    assert create_kwargs["default_feature_ids"] == ["web_search"]

    update_kwargs = client.update_model.await_args.kwargs
    assert update_kwargs["capabilities"] == {"web_search": True}
    assert update_kwargs["skill_ids"] == []


@pytest.mark.asyncio
async def test_model_handlers_forward_model_suggestion_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.create_model = AsyncMock(return_value={"id": "new-model"})
    client.update_model = AsyncMock(return_value={"id": "existing-model"})
    monkeypatch.setattr(main, "get_client", lambda: client)
    monkeypatch.setenv("MCP_PROFILE", "local")
    monkeypatch.setenv("OPENWEBUI_API_KEY", "session-token")
    suggestions = [main.PromptSuggestionParam(title=["Reflect", "on today"], content="Reflect now")]

    await main.create_model(
        main.ModelCreateParam(
            id="new-model",
            name="New Model",
            base_model_id="gpt-5",
            suggestion_prompts=suggestions,
        ),
        Mock(),
    )
    await main.update_model(
        main.ModelUpdateParam(model_id="existing-model", suggestion_prompts=suggestions),
        Mock(),
    )

    expected = [{"title": ["Reflect", "on today"], "content": "Reflect now"}]
    assert client.create_model.await_args.kwargs["meta"] == {"suggestion_prompts": expected}
    assert client.update_model.await_args.kwargs["meta"] == {"suggestion_prompts": expected}


def test_knowledge_access_tool_schema_exposes_native_grants() -> None:
    schema = main.KnowledgeAccessParam.model_json_schema()

    assert set(schema["properties"]) == {"knowledge_id", "access_grants"}
    assert schema["properties"]["access_grants"]["type"] == "array"


@pytest.mark.asyncio
async def test_update_knowledge_access_forwards_token_and_preserves_api_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    client.update_knowledge_access = AsyncMock(
        return_value={
            "id": "knowledge-1",
            "name": "Research",
            "description": "Existing description",
            "access_grants": [
                {"principal_type": "group", "principal_id": "research", "permission": "read"}
            ],
            "files": [],
        }
    )
    monkeypatch.setattr(main, "get_client", lambda: client)
    monkeypatch.setenv("MCP_PROFILE", "local")
    monkeypatch.setenv("OPENWEBUI_API_KEY", "session-token")
    grants = [{"principal_type": "group", "principal_id": "research", "permission": "read"}]

    result = await main.update_knowledge_access(
        main.KnowledgeAccessParam(knowledge_id="knowledge-1", access_grants=grants), Mock()
    )

    client.update_knowledge_access.assert_awaited_once_with("knowledge-1", grants, "session-token")
    assert result["id"] == "knowledge-1"
    assert result["access_grants"] == grants
    assert result["files"] == []
    assert result == {
        "id": "knowledge-1",
        "name": "Research",
        "description": "Existing description",
        "access_grants": grants,
        "files": [],
    }


@pytest.mark.asyncio
async def test_create_tool_schema_and_handler_forward_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = main.ToolCreateParam.model_json_schema()
    assert schema["properties"]["meta"]["anyOf"]

    client = Mock()
    client.create_tool = AsyncMock(return_value={"id": "weather_tool"})
    monkeypatch.setattr(main, "get_client", lambda: client)
    monkeypatch.setenv("MCP_PROFILE", "local")
    monkeypatch.setenv("OPENWEBUI_API_KEY", "session-token")

    result = await main.create_tool(
        main.ToolCreateParam(
            id="weather_tool",
            name="Weather",
            content="class Tools: pass",
            meta={"description": "Weather lookup"},
        ),
        Mock(),
    )

    client.create_tool.assert_awaited_once_with(
        "weather_tool",
        "Weather",
        "class Tools: pass",
        meta={"description": "Weather lookup"},
        api_key="session-token",
    )
    assert result == {"id": "weather_tool"}


async def test_log_mutation_logs_action_and_returns_result_unchanged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    result = {"id": "custom"}
    with caplog.at_level("INFO", logger="openwebui_mcp.main"):
        returned = main.log_mutation("model.update", "custom", ["name"], result)

    assert returned is result
    assert "model.update" in caplog.text
