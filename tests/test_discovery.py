from unittest.mock import AsyncMock

import httpx
import pytest

from openwebui_mcp.discovery import (
    build_environment_info,
    project_openwebui_version,
    validate_model_configuration,
)


def test_project_openwebui_version_extracts_supported_fields() -> None:
    assert project_openwebui_version(
        {"version": "0.7.2", "deployment_id": "dep-abc", "extra": "ignored"}
    ) == {
        "version": "0.7.2",
        "deployment_id": "dep-abc",
    }


def test_build_environment_info_reports_runtime_active_operations_only() -> None:
    payload = build_environment_info(
        {"version": "0.7.2", "deployment_id": "dep"},
        mcp_profile="member",
        active_operations=["get_model", "validate_model_configuration"],
    )

    assert payload["openwebui"] == {"version": "0.7.2", "deployment_id": "dep"}
    assert payload["mcp_profile"] == "member"
    assert payload["mcp_operations"]["active"] == [
        "get_model",
        "validate_model_configuration",
    ]
    assert "delete_model" not in payload["mcp_operations"]["active"]
    assert payload["mcp_operations"]["source"] == "runtime_profile"


@pytest.mark.asyncio
async def test_validate_model_configuration_reports_missing_model_and_caller_scope() -> None:
    client = AsyncMock()
    client._get_model_record_or_none = AsyncMock(return_value=None)

    result = await validate_model_configuration(client, "missing-model", "token")

    assert result["status"] == "error"
    assert result["validation_subject"]["scope"] == "current_caller"
    assert result["stored"] is None
    assert result["checks"][0]["id"] == "model_record"
    assert result["checks"][0]["status"] == "fail"


@pytest.mark.asyncio
async def test_validate_model_configuration_happy_path_without_credentials() -> None:
    client = AsyncMock()
    client._get_model_record_or_none = AsyncMock(
        return_value={
            "id": "custom",
            "name": "Custom",
            "base_model_id": "gpt-5",
            "is_active": True,
            "params": {"temperature": 0.4, "function_calling": "native"},
            "meta": {
                "toolIds": ["tool-a"],
                "skillIds": ["skill-a"],
                "capabilities": {"vision": True},
                "builtinTools": {"memory": False},
                "knowledge": [{"id": "kb-1", "type": "collection"}],
                "defaultFeatureIds": [],
                "apiKey": "must-not-leak",
            },
        }
    )
    client.list_models = AsyncMock(return_value={"data": [{"id": "gpt-5"}]})
    client.list_tools = AsyncMock(return_value={"data": [{"id": "tool-a"}]})
    client.get = AsyncMock(return_value={"id": "ok"})
    client.get_models_config = AsyncMock(return_value={"DEFAULT_MODELS": ["gpt-5"]})
    client._path_id = lambda value: value

    result = await validate_model_configuration(client, "custom", "token")

    assert result["status"] == "warning"
    assert "must-not-leak" not in str(result)
    assert result["stored"]["params"]["temperature"] == 0.4
    assert result["effective"]["function_calling"]["effective"]["source"] == "stored"
    assert result["effective"]["capabilities"]["stored_value"] == {"vision": True}
    assert result["effective"]["capabilities"]["effective"]["source"] == "unknown"
    assert "global" in result["effective"]["capabilities"]["effective"]["note"].casefold()


@pytest.mark.asyncio
async def test_validate_model_configuration_defaults_missing_function_calling_to_native() -> None:
    client = AsyncMock()
    client._get_model_record_or_none = AsyncMock(
        return_value={
            "id": "custom",
            "name": "Custom",
            "base_model_id": "gpt-5",
            "params": {},
            "meta": {"defaultFeatureIds": []},
        }
    )
    client.list_models = AsyncMock(return_value={"data": [{"id": "gpt-5"}]})
    client.list_tools = AsyncMock(return_value={"data": []})
    client.get_models_config = AsyncMock(return_value={})
    client._path_id = lambda value: value

    result = await validate_model_configuration(client, "custom", "token")

    fc = result["effective"]["function_calling"]
    assert fc["stored_value"] is None
    assert fc["effective"]["source"] == "openwebui_default"
    assert fc["effective"]["value"] == "native"
    assert any(c["id"] == "function_calling" and c["status"] == "pass" for c in result["checks"])


@pytest.mark.asyncio
async def test_validate_model_configuration_rejects_unsupported_function_calling() -> None:
    client = AsyncMock()
    client._get_model_record_or_none = AsyncMock(
        return_value={
            "id": "custom",
            "name": "Custom",
            "base_model_id": "gpt-5",
            "params": {"function_calling": "compat"},
            "meta": {"defaultFeatureIds": []},
        }
    )
    client.list_models = AsyncMock(return_value={"data": [{"id": "gpt-5"}]})
    client.list_tools = AsyncMock(return_value={"data": []})
    client.get_models_config = AsyncMock(return_value={})
    client._path_id = lambda value: value

    result = await validate_model_configuration(client, "custom", "token")

    assert result["status"] == "error"
    assert any(c["id"] == "function_calling" and c["status"] == "fail" for c in result["checks"])


@pytest.mark.asyncio
async def test_validate_model_configuration_flags_missing_and_inaccessible_refs() -> None:
    client = AsyncMock()
    client._get_model_record_or_none = AsyncMock(
        return_value={
            "id": "custom",
            "name": "Custom",
            "base_model_id": "missing-base",
            "params": {},
            "meta": {
                "toolIds": ["missing-tool"],
                "skillIds": ["private-skill", "ghost-skill"],
                "knowledge": [{"id": "ghost-kb", "type": "collection"}],
            },
        }
    )
    client.list_models = AsyncMock(return_value={"data": [{"id": "gpt-5"}]})
    client.list_tools = AsyncMock(return_value={"data": []})
    client.get_models_config = AsyncMock(side_effect=_http_error(401))
    client._path_id = lambda value: value

    async def skill_get(path: str, api_key: str) -> dict:
        if path.endswith("/private-skill"):
            raise _http_error(401)
        raise _http_error(404)

    client.get = AsyncMock(side_effect=skill_get)

    result = await validate_model_configuration(client, "custom", "token")

    assert result["status"] == "error"
    assert result["effective"]["capabilities"]["effective"]["source"] == "unknown"
    assert any(
        c["id"] == "instance_defaults" and c["status"] == "unknown" for c in result["checks"]
    )


@pytest.mark.asyncio
async def test_validate_model_configuration_unknown_effective_gates_warn() -> None:
    client = AsyncMock()
    client._get_model_record_or_none = AsyncMock(
        return_value={
            "id": "custom",
            "name": "Custom",
            "base_model_id": "gpt-5",
            "params": {},
            "meta": {},
        }
    )
    client.list_models = AsyncMock(return_value={"data": [{"id": "gpt-5"}]})
    client.list_tools = AsyncMock(return_value={"data": []})
    client.get_models_config = AsyncMock(return_value={})
    client._path_id = lambda value: value

    result = await validate_model_configuration(client, "custom", "token")

    assert result["status"] == "warning"
    assert result["effective"]["builtin_tools"]["effective"]["source"] == "unknown"
    assert result["effective"]["function_calling"]["effective"]["source"] == "openwebui_default"


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://webui.example/resource")
    response = httpx.Response(status, request=request, text="{}")
    return httpx.HTTPStatusError(str(status), request=request, response=response)
