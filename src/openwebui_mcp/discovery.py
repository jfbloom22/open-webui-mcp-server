"""Read-only environment discovery and custom-model configuration validation."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Awaitable, Callable, Literal

import httpx

from .security import sanitize_value

CheckStatus = Literal["pass", "fail", "warning", "unknown"]

MCP_CAPABILITIES: tuple[str, ...] = (
    "credential_safe_projections",
    "environment_discovery",
    "member_profile_tool_scoping",
    "model_configuration_validation",
    "openwebui_management_api",
    "saved_model_generation_smoke_test",
)

SUPPORTED_FUNCTION_CALLING_MODES = frozenset({"native", "legacy"})
OPENWEBUI_FUNCTION_CALLING_DEFAULT = "native"
OPENWEBUI_FUNCTION_CALLING_DEFAULT_NOTE = (
    "Open WebUI resolves omitted params.function_calling to 'native' when building "
    "chat metadata (backend/open_webui/main.py)."
)

_UNKNOWN_INHERIT_NOTE = (
    "Effective runtime value may inherit from the base model or instance defaults; "
    "not computed by this read-only check."
)

_GATEABLE_META_NOTE = (
    "stored_value is taken from the saved model record only. Global Open WebUI "
    "feature gates may still block runtime use; effective globally-enabled state "
    "is not computed."
)

_CALLER_SCOPE_NOTE = (
    "Checks use the authenticated Open WebUI session forwarded to this MCP server. "
    "Results describe what the current caller can read and use, not another user's access."
)


def mcp_package_version() -> str:
    try:
        return version("openwebui-mcp-server")
    except PackageNotFoundError:
        from . import __version__

        return __version__


def project_openwebui_version(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract version fields from GET /api/version (not config export)."""
    if not isinstance(payload, dict):
        return {"version": None, "deployment_id": None}
    return {
        "version": payload.get("version"),
        "deployment_id": payload.get("deployment_id"),
    }


def build_environment_info(
    openwebui_version_payload: dict[str, Any],
    *,
    mcp_profile: str,
    active_operations: list[str],
) -> dict[str, Any]:
    """Build MCP environment discovery payload without config.version inference."""
    return {
        "openwebui": project_openwebui_version(openwebui_version_payload),
        "mcp_package": {
            "name": "openwebui-mcp-server",
            "version": mcp_package_version(),
        },
        "mcp_profile": mcp_profile,
        "mcp_capabilities": list(MCP_CAPABILITIES),
        "mcp_operations": {
            "source": "runtime_profile",
            "active": sorted(active_operations),
            "note": (
                "MCP tools registered for this server process after MCP_PROFILE and "
                "OPENWEBUI_DISABLED_TOOLS filtering."
            ),
        },
    }


def _check(
    check_id: str,
    status: CheckStatus,
    message: str,
    *,
    resource_id: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"id": check_id, "status": status, "message": message}
    if resource_id is not None:
        item["resource_id"] = resource_id
    return item


def _aggregate_status(checks: list[dict[str, Any]]) -> Literal["ok", "warning", "error"]:
    if any(item.get("status") == "fail" for item in checks):
        return "error"
    if any(item.get("status") in {"warning", "unknown"} for item in checks):
        return "warning"
    return "ok"


def _gated_meta_field(stored_value: Any) -> dict[str, Any]:
    if stored_value is None:
        return {
            "stored_value": None,
            "effective": {"source": "unknown", "value": None, "note": _UNKNOWN_INHERIT_NOTE},
        }
    return {
        "stored_value": stored_value,
        "effective": {"source": "unknown", "value": None, "note": _GATEABLE_META_NOTE},
    }


def _list_binding_field(stored_value: Any) -> dict[str, Any]:
    if stored_value is None:
        return {
            "stored_value": None,
            "effective": {"source": "unknown", "value": None, "note": _UNKNOWN_INHERIT_NOTE},
        }
    return {
        "stored_value": stored_value,
        "effective": {
            "source": "stored",
            "value": stored_value,
            "note": "List bindings as stored; per-item access is validated in checks.",
        },
    }


def _function_calling_field(stored_value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if stored_value is None:
        return (
            _check(
                "function_calling",
                "pass",
                "Stored value omitted; Open WebUI default is native",
            ),
            {
                "stored_value": None,
                "effective": {
                    "source": "openwebui_default",
                    "value": OPENWEBUI_FUNCTION_CALLING_DEFAULT,
                    "note": OPENWEBUI_FUNCTION_CALLING_DEFAULT_NOTE,
                },
            },
        )
    if stored_value in SUPPORTED_FUNCTION_CALLING_MODES:
        return (
            _check(
                "function_calling",
                "pass",
                f"Stored params.function_calling is supported ({stored_value})",
            ),
            {
                "stored_value": stored_value,
                "effective": {
                    "source": "stored",
                    "value": stored_value,
                    "note": "Matches stored params.function_calling.",
                },
            },
        )
    return (
        _check(
            "function_calling",
            "fail",
            f"Unsupported stored params.function_calling value: {stored_value!r}",
        ),
        {
            "stored_value": stored_value,
            "effective": {
                "source": "unknown",
                "value": None,
                "note": "Unsupported stored value; supported modes are native and legacy.",
            },
        },
    )


def _stored_model_projection(model: dict[str, Any]) -> dict[str, Any]:
    meta = model.get("meta") if isinstance(model.get("meta"), dict) else {}
    knowledge_refs: list[dict[str, str]] = []
    if isinstance(meta.get("knowledge"), list):
        for item in meta["knowledge"]:
            if not isinstance(item, dict):
                continue
            ref_id = item.get("id")
            ref_type = item.get("type")
            if isinstance(ref_id, str) and isinstance(ref_type, str):
                knowledge_refs.append({"id": ref_id, "type": ref_type})
    projected = {
        "id": model.get("id"),
        "name": model.get("name"),
        "base_model_id": model.get("base_model_id"),
        "is_active": model.get("is_active"),
        "params": model.get("params") if isinstance(model.get("params"), dict) else {},
        "meta": {
            "description": meta.get("description"),
            "toolIds": meta.get("toolIds"),
            "skillIds": meta.get("skillIds"),
            "defaultFeatureIds": meta.get("defaultFeatureIds"),
            "capabilities": meta.get("capabilities"),
            "builtinTools": meta.get("builtinTools"),
            "knowledge": knowledge_refs or None,
        },
    }
    sanitized = sanitize_value(projected)
    return sanitized if isinstance(sanitized, dict) else {}


async def _probe_get(
    fetch: Callable[[], Awaitable[Any]],
) -> tuple[CheckStatus, str]:
    try:
        await fetch()
        return "pass", "ok"
    except httpx.HTTPStatusError as error:
        status_code = error.response.status_code if error.response is not None else None
        if status_code == 404:
            return "fail", "not found"
        if status_code == 401:
            return "fail", "inaccessible"
        return "fail", "request failed"
    except ValueError:
        return "fail", "not found"


async def validate_model_configuration(
    client: Any,
    model_id: str,
    api_key: str | None,
) -> dict[str, Any]:
    """Validate a saved custom model and report stored settings (read-only)."""
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    validation_subject = {"scope": "current_caller", "note": _CALLER_SCOPE_NOTE}

    model = await client._get_model_record_or_none(model_id, api_key)
    if not model:
        checks.append(
            _check(
                "model_record",
                "fail",
                "Custom model record not found or not accessible to the caller",
            )
        )
        return {
            "model_id": model_id,
            "validation_subject": validation_subject,
            "status": "error",
            "checks": checks,
            "warnings": warnings,
            "stored": None,
            "effective": {},
            "instance_defaults": {
                "source": "not_loaded",
                "available": False,
                "note": "Model record could not be loaded.",
            },
        }

    checks.append(_check("model_record", "pass", "Custom model record is readable"))
    stored = _stored_model_projection(model)
    meta = model.get("meta") if isinstance(model.get("meta"), dict) else {}
    params = model.get("params") if isinstance(model.get("params"), dict) else {}

    base_model_id = model.get("base_model_id")
    accessible_ids: set[str] = set()
    if isinstance(base_model_id, str) and base_model_id:
        catalog = await client.list_models(api_key=api_key)
        for item in catalog.get("data") or []:
            if isinstance(item, dict) and item.get("id"):
                accessible_ids.add(str(item["id"]))
        if base_model_id in accessible_ids:
            checks.append(
                _check(
                    "base_model",
                    "pass",
                    "Base model is present in the caller's effective model catalog",
                    resource_id=base_model_id,
                )
            )
        else:
            checks.append(
                _check(
                    "base_model",
                    "fail",
                    "Base model is not available in the caller's effective model catalog",
                    resource_id=base_model_id,
                )
            )
    else:
        checks.append(_check("base_model", "fail", "Saved model is missing base_model_id"))

    fc_check, effective_function_calling = _function_calling_field(params.get("function_calling"))
    checks.append(fc_check)

    tools_payload = await client.list_tools(api_key)
    tools_key = next(
        (key for key in ("data", "items", "tools") if isinstance(tools_payload.get(key), list)),
        None,
    )
    known_tools = {
        str(item["id"])
        for item in (tools_payload.get(tools_key, []) if tools_key else [])
        if isinstance(item, dict) and item.get("id")
    }
    tool_ids = meta.get("toolIds") if isinstance(meta.get("toolIds"), list) else []
    for tool_id in tool_ids:
        if not isinstance(tool_id, str):
            continue
        if tool_id in known_tools:
            checks.append(
                _check(
                    "tool_reference", "pass", "Tool is listed for the caller", resource_id=tool_id
                )
            )
        else:
            checks.append(
                _check(
                    "tool_reference",
                    "fail",
                    "Tool ID is missing from the caller's tool list",
                    resource_id=tool_id,
                )
            )

    skill_ids = meta.get("skillIds") if isinstance(meta.get("skillIds"), list) else []
    for skill_id in skill_ids:
        if not isinstance(skill_id, str):
            continue
        status, reason = await _probe_get(
            lambda skill_id=skill_id: client.get(
                f"/api/v1/skills/id/{client._path_id(skill_id)}",
                api_key,
            )
        )
        if status == "pass":
            checks.append(
                _check(
                    "skill_reference",
                    "pass",
                    "Skill is readable for the caller",
                    resource_id=skill_id,
                )
            )
        elif reason == "inaccessible":
            checks.append(
                _check(
                    "skill_reference",
                    "fail",
                    "Skill exists but is not accessible to the caller",
                    resource_id=skill_id,
                )
            )
        else:
            checks.append(
                _check(
                    "skill_reference",
                    "fail",
                    "Skill ID was not found",
                    resource_id=skill_id,
                )
            )

    knowledge = meta.get("knowledge") if isinstance(meta.get("knowledge"), list) else []
    for ref in knowledge:
        if not isinstance(ref, dict) or ref.get("type") != "collection":
            continue
        collection_id = ref.get("id")
        if not isinstance(collection_id, str):
            continue
        status, reason = await _probe_get(
            lambda collection_id=collection_id: client.get(
                f"/api/v1/knowledge/{client._path_id(collection_id)}",
                api_key,
            )
        )
        if status == "pass":
            checks.append(
                _check(
                    "knowledge_reference",
                    "pass",
                    "Knowledge collection is readable for the caller",
                    resource_id=collection_id,
                )
            )
        elif reason == "inaccessible":
            checks.append(
                _check(
                    "knowledge_reference",
                    "fail",
                    "Knowledge collection is not accessible to the caller",
                    resource_id=collection_id,
                )
            )
        else:
            checks.append(
                _check(
                    "knowledge_reference",
                    "fail",
                    "Knowledge collection was not found",
                    resource_id=collection_id,
                )
            )

    instance_defaults: dict[str, Any] = {
        "source": "instance_config",
        "available": False,
        "fields": {},
        "note": (
            "Allowlisted /api/v1/configs/models fields may influence defaults; "
            "not merged into effective values here."
        ),
    }
    try:
        instance_defaults["fields"] = await client.get_models_config(api_key)
        instance_defaults["available"] = True
        checks.append(
            _check(
                "instance_defaults",
                "pass",
                "Instance model defaults config is readable (allowlisted fields only)",
            )
        )
    except httpx.HTTPStatusError:
        instance_defaults["available"] = False
        checks.append(
            _check(
                "instance_defaults",
                "unknown",
                "Instance model defaults config is not available to the caller",
            )
        )
        warnings.append(
            "Instance default model configuration could not be read; gating is unknown."
        )

    effective = {
        "function_calling": effective_function_calling,
        "capabilities": _gated_meta_field(meta.get("capabilities")),
        "builtin_tools": _gated_meta_field(meta.get("builtinTools")),
        "skill_ids": _list_binding_field(meta.get("skillIds")),
        "default_feature_ids": _gated_meta_field(meta.get("defaultFeatureIds")),
    }
    if any(
        entry.get("effective", {}).get("source") == "unknown"
        for entry in effective.values()
        if isinstance(entry, dict)
    ):
        warnings.append(_UNKNOWN_INHERIT_NOTE)
    if any(
        entry.get("effective", {}).get("note") == _GATEABLE_META_NOTE
        for entry in effective.values()
        if isinstance(entry, dict)
    ):
        warnings.append(_GATEABLE_META_NOTE)

    overall = _aggregate_status(checks)
    if overall == "ok" and warnings:
        overall = "warning"

    return sanitize_value(
        {
            "model_id": model_id,
            "validation_subject": validation_subject,
            "status": overall,
            "checks": checks,
            "warnings": warnings,
            "stored": stored,
            "effective": effective,
            "instance_defaults": instance_defaults,
        }
    )
