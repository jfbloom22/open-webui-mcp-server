"""Redact secrets and bulky file payloads from MCP-facing responses and errors."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._\-+/=]+", re.IGNORECASE)
_SENSITIVE_KEY_RE = re.compile(
    r"(^|[_\.])("
    r"api[_-]?keys?|access[_-]?keys?|private[_-]?keys?|subscription[_-]?keys?|"
    r"secrets?|passwords?|tokens?|credentials?|"
    r"auth(?:entication|_token)?|cookies?|headers?|keys?"
    r")([_\.]|$)",
    re.IGNORECASE,
)
_SENSITIVE_KEY_SEGMENTS = frozenset(
    {
        "key",
        "keys",
        "secret",
        "secrets",
        "password",
        "passwords",
        "token",
        "tokens",
        "credential",
        "credentials",
        "auth",
        "authorization",
        "cookie",
        "cookies",
        "header",
        "headers",
    }
)
_NON_SENSITIVE_NORMALIZED_KEYS = frozenset(
    {
        "max_tokens",
        "token_type",
    }
)
_CREDENTIAL_QUERY_ALIASES = frozenset(
    {
        "api-key",
        "api_key",
        "apikey",
        "key",
        "token",
        "access_token",
        "auth",
        "auth_token",
        "client_secret",
        "secret",
        "password",
        "x_api_key",
    }
)
_REDACTED_URL = "[redacted-url]"
_GENERIC_ERROR_DETAIL = "Request failed."
_MAX_SAFE_STRING_CHARS = 2000
_MAX_CONFIG_LIST_ITEMS = 200

CURRENT_USER_FIELDS: tuple[str, ...] = (
    "id",
    "email",
    "name",
    "role",
    "permissions",
    "profile_image_url",
    "bio",
    "gender",
    "date_of_birth",
    "status_emoji",
    "status_message",
    "status_expires_at",
    "expires_at",
    "token_type",
)

FILE_SEARCH_RECORD_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "filename",
        "knowledge_ids",
        "status",
        "content_type",
        "size",
    }
)

FILE_DETAIL_EXTRA_KEYS: tuple[str, ...] = (
    "user_id",
    "hash",
    "created_at",
    "updated_at",
)

# Top-level export keys safe to return to MCP clients (unknown export keys are omitted).
CONFIG_EXPORT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ui.banners",
        "ui.prompt_suggestions",
        "ui.default_locale",
        "ui.default_models",
        "DEFAULT_MODELS",
        "DEFAULT_PINNED_MODELS",
        "MODEL_ORDER_LIST",
        "models.default_metadata",
        "ENABLE_OPENAI_API",
        "ENABLE_OLLAMA_API",
        "ENABLE_DIRECT_CONNECTIONS",
        "ENABLE_BASE_MODELS_CACHE",
        "ENABLE_SIGNUP",
        "ENABLE_LOGIN_FORM",
        "ENABLE_API_KEYS",
        "ENABLE_COMMUNITY_SHARING",
        "ENABLE_MESSAGE_RATING",
        "ENABLE_ADMIN_CHAT_ACCESS",
        "ENABLE_ADMIN_EXPORT",
    }
)

CONFIG_EXPORT_PRESENCE_KEYS: frozenset[str] = frozenset(
    {
        "openai.api_keys",
    }
)

CONFIG_EXPORT_URL_LIST_KEYS: frozenset[str] = frozenset(
    {
        "openai.api_base_urls",
    }
)

MODELS_CONFIG_ALLOWLIST: frozenset[str] = frozenset(
    {
        "DEFAULT_MODELS",
        "DEFAULT_PINNED_MODELS",
        "MODEL_ORDER_LIST",
        "DEFAULT_MODEL_METADATA",
        "DEFAULT_MODEL_PARAMS",
    }
)


def secret_presence(value: Any) -> dict[str, bool]:
    """Indicate whether a secret exists without fragments, lengths, or key names."""
    if value is None:
        return {"present": False}
    if isinstance(value, str):
        return {"present": bool(value)}
    if isinstance(value, (list, dict, tuple, set)):
        return {"present": len(value) > 0}
    return {"present": True}


def _is_presence_marker(value: Any) -> bool:
    return isinstance(value, dict) and set(value) <= {"present"}


def _normalize_sensitive_key_name(key: str) -> str:
    text = key.replace("-", "_").replace(".", "_")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return text.lower()


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in {"authorization", "cookie", "set-cookie"}:
        return True
    normalized = _normalize_sensitive_key_name(key)
    if normalized in _NON_SENSITIVE_NORMALIZED_KEYS:
        return False
    if _SENSITIVE_KEY_RE.search(normalized):
        return True
    segments = [part for part in normalized.split("_") if part]
    return any(part in _SENSITIVE_KEY_SEGMENTS for part in segments)


def _is_credential_query_param(name: str) -> bool:
    normalized = _normalize_sensitive_key_name(name)
    if normalized in _CREDENTIAL_QUERY_ALIASES:
        return True
    return _is_sensitive_key(name)


def _safe_query_pairs(query: str) -> list[tuple[str, str]]:
    return [
        (name, value)
        for name, value in parse_qsl(query, keep_blank_values=True)
        if not _is_credential_query_param(name)
    ]


def _raw_url_looks_sensitive(raw: str) -> bool:
    if "@" in raw:
        return True
    if "?" not in raw:
        return False
    _, _, query = raw.partition("?")
    return any(_is_credential_query_param(name) for name, _ in parse_qsl(query))


def sanitize_connection_path(path: str | None) -> str | None:
    """Return a path without embedded queries, fragments, or userinfo."""
    if not isinstance(path, str) or not path.strip():
        return path
    text = path.strip().split("#", 1)[0]
    if _raw_url_looks_sensitive(text):
        without_query = text.split("?", 1)[0]
        return without_query.split("@")[-1] or "/"
    if "?" in text:
        base, _, query = text.partition("?")
        safe = urlencode(_safe_query_pairs(query))
        rebuilt = f"{base}?{safe}" if safe else base
        return rebuilt.split("@")[-1]
    return text.split("@")[-1]


def sanitize_service_url(url: str | None) -> str | None:
    """Return origin-style URL without credentials, fragments, or sensitive query values."""
    if not isinstance(url, str) or not url.strip():
        return url
    raw = url.strip()
    parsed = urlparse(raw)
    if not parsed.scheme:
        if _raw_url_looks_sensitive(raw):
            return _REDACTED_URL
        if "?" in raw:
            base, _, query = raw.partition("?")
            safe = urlencode(_safe_query_pairs(query))
            return f"{base}?{safe}" if safe else base.split("@")[-1]
        return raw.split("@")[-1] if "@" not in raw else _REDACTED_URL
    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunparse(
        (
            parsed.scheme,
            host,
            parsed.path or "",
            parsed.params,
            urlencode(_safe_query_pairs(parsed.query)),
            "",
        )
    )


def _copy_safe_config_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        if len(value) > _MAX_SAFE_STRING_CHARS:
            return value[: _MAX_SAFE_STRING_CHARS - 1] + "…"
        return value
    if isinstance(value, list):
        return [_copy_safe_config_value(item) for item in value[:_MAX_CONFIG_LIST_ITEMS]]
    if isinstance(value, dict):
        copied: dict[str, Any] = {}
        for key, item in list(value.items())[:100]:
            if _is_sensitive_key(str(key)):
                copied[str(key)] = secret_presence(item)
            else:
                copied[str(key)] = _copy_safe_config_value(item)
        return copied
    return None


def project_config_export(config: dict[str, Any]) -> dict[str, Any]:
    """Return only allowlisted export keys; never pass unknown configuration through."""
    if not isinstance(config, dict):
        return {}
    projected: dict[str, Any] = {}
    for key in CONFIG_EXPORT_ALLOWLIST:
        if key not in config:
            continue
        projected[key] = _copy_safe_config_value(config[key])
    for key in CONFIG_EXPORT_PRESENCE_KEYS:
        if key in config:
            projected[key] = secret_presence(config[key])
    for key in CONFIG_EXPORT_URL_LIST_KEYS:
        if key in config:
            urls = config[key]
            if isinstance(urls, list):
                projected[key] = [
                    sanitize_service_url(item) for item in urls if isinstance(item, str)
                ]
    return projected


def project_models_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Project the /api/v1/configs/models response to known safe fields."""
    if not isinstance(payload, dict):
        return {}
    projected: dict[str, Any] = {}
    for key in MODELS_CONFIG_ALLOWLIST:
        if key not in payload:
            continue
        projected[key] = _copy_safe_config_value(payload[key])
    return projected


def sanitize_value(value: Any) -> Any:
    """Recursively redact sensitive values in structured data (errors and nested dicts)."""
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _is_presence_marker(item):
                sanitized[key] = item
            elif _is_sensitive_key(str(key)):
                sanitized[key] = secret_presence(item)
            else:
                sanitized[key] = sanitize_value(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


def sanitize_current_user(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose session profile fields without authentication tokens or API secrets."""
    if not isinstance(payload, dict):
        return {}
    projected = {key: payload[key] for key in CURRENT_USER_FIELDS if key in payload}
    permissions = projected.get("permissions")
    if isinstance(permissions, dict):
        projected["permissions"] = sanitize_value(permissions)
    return projected


def _project_tool_server_connection(connection: dict[str, Any]) -> dict[str, Any]:
    info = connection.get("info") if isinstance(connection.get("info"), dict) else {}
    config = connection.get("config") if isinstance(connection.get("config"), dict) else {}
    projected = {
        "url": sanitize_service_url(connection.get("url")),
        "path": sanitize_connection_path(connection.get("path")),
        "type": connection.get("type"),
        "auth_type": connection.get("auth_type"),
        "forward_cookies": connection.get("forward_cookies"),
        "info": {
            key: info.get(key)
            for key in ("id", "name", "description", "title")
            if info.get(key) is not None
        },
        "config": {
            key: config.get(key) for key in ("enable", "function_name_filter_list") if key in config
        },
    }
    if connection.get("key") is not None:
        projected["key"] = secret_presence(connection.get("key"))
    if connection.get("headers") is not None:
        projected["headers"] = secret_presence(connection.get("headers"))
    return projected


def sanitize_tool_servers_config(config: dict[str, Any]) -> dict[str, Any]:
    """Expose tool-server connection metadata without auth material."""
    connections = config.get("TOOL_SERVER_CONNECTIONS", [])
    if not isinstance(connections, list):
        return {"TOOL_SERVER_CONNECTIONS": []}
    return {
        "TOOL_SERVER_CONNECTIONS": [
            _project_tool_server_connection(item) for item in connections if isinstance(item, dict)
        ]
    }


def _strip_content_from_mapping(item: dict[str, Any]) -> dict[str, Any]:
    result = dict(item)
    data = result.get("data")
    if isinstance(data, dict) and "content" in data:
        trimmed = dict(data)
        trimmed.pop("content", None)
        result["data"] = trimmed
    meta = result.get("meta")
    if isinstance(meta, dict):
        meta_data = meta.get("data")
        if isinstance(meta_data, dict) and "content" in meta_data:
            trimmed_meta = dict(meta)
            trimmed_data = dict(meta_data)
            trimmed_data.pop("content", None)
            trimmed_meta["data"] = trimmed_data
            result["meta"] = trimmed_meta
    return result


def strip_file_content_fields(value: Any) -> Any:
    """Remove extracted document content from file records."""
    if isinstance(value, list):
        return [strip_file_content_fields(item) for item in value]
    if isinstance(value, dict):
        if "items" in value and isinstance(value["items"], list):
            return {**value, "items": [strip_file_content_fields(item) for item in value["items"]]}
        if "data" in value and isinstance(value["data"], list):
            return {**value, "data": [strip_file_content_fields(item) for item in value["data"]]}
        return _strip_content_from_mapping(value)
    return value


def project_file_search_record(item: dict[str, Any]) -> dict[str, Any]:
    """Allowlisted metadata for filename search results (no raw meta/data blobs)."""
    trimmed = _strip_content_from_mapping(item)
    meta = trimmed.get("meta") if isinstance(trimmed.get("meta"), dict) else {}
    data = trimmed.get("data") if isinstance(trimmed.get("data"), dict) else {}
    meta_data = meta.get("data") if isinstance(meta.get("data"), dict) else {}

    knowledge_ids: list[str] = []
    for candidate in (
        meta_data.get("knowledge_id"),
        trimmed.get("knowledge_id"),
        trimmed.get("collection_name"),
    ):
        if isinstance(candidate, str) and candidate and candidate not in knowledge_ids:
            knowledge_ids.append(candidate)

    content_type = meta.get("content_type")
    if isinstance(content_type, list):
        content_type = next((value for value in content_type if isinstance(value, str)), None)
    elif content_type is not None and not isinstance(content_type, str):
        content_type = None

    record: dict[str, Any] = {
        "id": trimmed.get("id"),
        "filename": trimmed.get("filename"),
    }
    if knowledge_ids:
        record["knowledge_ids"] = knowledge_ids
    status = data.get("status")
    if status is not None:
        record["status"] = status
    if content_type is not None:
        record["content_type"] = content_type
    size = meta.get("size")
    if isinstance(size, int):
        record["size"] = size
    return {
        key: record[key]
        for key in FILE_SEARCH_RECORD_KEYS
        if key in record and record[key] is not None
    }


def project_file_detail_record(item: dict[str, Any]) -> dict[str, Any]:
    """Content-free file metadata for get/list, including stable listing fields."""
    record = project_file_search_record(item)
    trimmed = _strip_content_from_mapping(item)
    for key in FILE_DETAIL_EXTRA_KEYS:
        if trimmed.get(key) is not None:
            record[key] = trimmed[key]
    return record


def compact_file_search_collection(payload: dict[str, Any]) -> dict[str, Any]:
    """Project search response lists to the narrow search allowlist."""
    for key in ("items", "data"):
        items = payload.get(key)
        if isinstance(items, list):
            return {
                **{name: value for name, value in payload.items() if name != key},
                key: [project_file_search_record(item) for item in items if isinstance(item, dict)],
            }
    if isinstance(payload, list):
        return {
            "data": [project_file_search_record(item) for item in payload if isinstance(item, dict)]
        }
    return project_file_search_record(payload) if isinstance(payload, dict) else payload


FILE_CONTENT_DEFAULT_START_LINE = 1
FILE_CONTENT_DEFAULT_MAX_LINES = 200
FILE_CONTENT_HARD_MAX_LINES = 500

KB_META_PUBLIC_KEYS: frozenset[str] = frozenset({"source", "read_only"})

KNOWLEDGE_DETAIL_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "name",
        "description",
        "user_id",
        "write_access",
        "created_at",
        "updated_at",
        "file_count",
    }
)

KNOWLEDGE_DIRECTORY_KEYS: frozenset[str] = frozenset({"id", "name", "parent_id", "knowledge_id"})


def project_knowledge_detail(payload: dict[str, Any]) -> dict[str, Any]:
    """Return KB metadata without embedded file records that may carry content."""
    if not isinstance(payload, dict):
        return {}
    files = payload.get("files")
    file_count = payload.get("file_count")
    if isinstance(files, list) and file_count is None:
        file_count = len(files)
    record: dict[str, Any] = {key: payload[key] for key in KNOWLEDGE_DETAIL_KEYS if key in payload}
    if file_count is not None:
        record["file_count"] = file_count
    meta = payload.get("meta")
    if isinstance(meta, dict):
        safe_meta = {key: meta[key] for key in KB_META_PUBLIC_KEYS if key in meta}
        external = meta.get("external")
        if isinstance(external, dict):
            safe_meta["external"] = {key: external[key] for key in ("provider",) if key in external}
        if safe_meta:
            record["meta"] = safe_meta
    grants = payload.get("access_grants")
    if isinstance(grants, list):
        record["access_grant_count"] = len(grants)
    record["file_listing"] = {
        "tool": "list_knowledge_files",
        "note": (
            "Embedded file arrays from GET /api/v1/knowledge/{id} are omitted here; "
            "use list_knowledge_files for metadata-only pagination."
        ),
    }
    sanitized = sanitize_value(record)
    return sanitized if isinstance(sanitized, dict) else record


def extract_file_content_text(payload: Any) -> str:
    """Normalize GET /api/v1/files/{id}/data/content responses."""
    if isinstance(payload, dict):
        for key in ("content", "text"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        data = payload.get("data")
        if isinstance(data, dict) and isinstance(data.get("content"), str):
            return data["content"]
    return payload if isinstance(payload, str) else ""


def project_bounded_file_content(
    payload: Any,
    *,
    start_line: int,
    max_lines: int,
) -> dict[str, Any]:
    """Return at most FILE_CONTENT_HARD_MAX_LINES from extracted file content."""
    bounded_max = max(1, min(max_lines, FILE_CONTENT_HARD_MAX_LINES))
    start = max(1, start_line)
    text = extract_file_content_text(payload)
    lines = text.splitlines()
    total = len(lines)
    if total == 0 or start > total:
        return {
            "content": "",
            "start_line": start,
            "end_line": 0,
            "line_count": 0,
            "total_lines": total,
            "has_more": False,
        }
    start_idx = start - 1
    end_idx = min(total, start_idx + bounded_max)
    chunk = lines[start_idx:end_idx]
    return {
        "content": "\n".join(chunk),
        "start_line": start,
        "end_line": end_idx,
        "line_count": len(chunk),
        "total_lines": total,
        "has_more": end_idx < total,
    }


def project_knowledge_file_list_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Project GET /api/v1/knowledge/{id}/files without extracted content."""
    if not isinstance(payload, dict):
        return {"items": [], "total": 0}
    items = payload.get("items")
    projected_items = [project_file_detail_record(item) for item in items if isinstance(item, dict)]
    result: dict[str, Any] = {
        "items": projected_items,
        "total": payload.get("total", len(projected_items)),
    }
    directories = payload.get("directories")
    if isinstance(directories, list):
        result["directories"] = [
            {key: entry[key] for key in KNOWLEDGE_DIRECTORY_KEYS if key in entry}
            for entry in directories
            if isinstance(entry, dict)
        ]
    breadcrumbs = payload.get("breadcrumbs")
    if isinstance(breadcrumbs, list):
        result["breadcrumbs"] = [
            {key: entry[key] for key in KNOWLEDGE_DIRECTORY_KEYS if key in entry}
            for entry in breadcrumbs
            if isinstance(entry, dict)
        ]
    return result


def project_file_detail_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip content and project a single file or paginated list for get/list paths."""
    if isinstance(payload, dict):
        for key in ("items", "data"):
            items = payload.get(key)
            if isinstance(items, list):
                return {
                    **payload,
                    key: [
                        project_file_detail_record(item) for item in items if isinstance(item, dict)
                    ],
                }
        return project_file_detail_record(payload)
    return strip_file_content_fields(payload)


def sanitize_error_detail(body: str, *, request_token: str | None = None) -> str:
    """Return a generic error detail; never echo arbitrary upstream body text."""
    _ = body, request_token
    return _GENERIC_ERROR_DETAIL


# Back-compat alias used by the client import.
sanitize_config_export = project_config_export
