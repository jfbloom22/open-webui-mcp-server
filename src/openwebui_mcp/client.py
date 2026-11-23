"""Open WebUI API client using a locally configured management credential."""

import asyncio
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote

import httpx

from .discovery import build_environment_info, validate_model_configuration
from .model_generation import run_saved_model_generation_test
from .security import (
    compact_file_search_collection,
    project_bounded_file_content,
    project_config_export,
    project_file_detail_response,
    project_knowledge_detail,
    project_knowledge_file_list_response,
    project_models_config,
    sanitize_current_user,
    sanitize_error_detail,
    sanitize_service_url,
    sanitize_tool_servers_config,
    strip_file_content_fields,
)

KNOWLEDGE_PAGE_SIZE = 30
KNOWLEDGE_FILES_LIST_PAGE_SIZE = 30
KNOWLEDGE_FILE_LIST_SCAN_MAX_PAGES = 34
_MEMBERSHIP_SCAN_LIMIT_MSG = (
    "Knowledge file membership could not be fully verified within the configured "
    "page scan limit (~{cap} files). The file may still be attached; paginate "
    "list_knowledge_files or retry with a smaller collection."
)


def _knowledge_file_scan_max_pages() -> int:
    """Safety cap for KB file listing scans (membership checks, scoped search)."""
    raw = os.getenv("OPENWEBUI_KB_FILE_SCAN_MAX_PAGES", "").strip()
    if raw.isdigit():
        return max(1, int(raw))
    return KNOWLEDGE_FILE_LIST_SCAN_MAX_PAGES


FILE_PROCESSING_WAIT_MAX_SECONDS = 30
FILE_PROCESSING_POLL_INTERVAL_SECONDS = 0.5
FILE_UPLOAD_BACKGROUND_HTTP_TIMEOUT_SECONDS = 60.0
FILE_UPLOAD_SYNC_HTTP_TIMEOUT_SECONDS = 120.0
DEFAULT_HTTP_TIMEOUT_SECONDS = 60.0
FILE_SEARCH_DEFAULT_LIMIT = 100
FILE_SEARCH_MAX_LIMIT = 1000


class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


UNSET = _UnsetType()


class OpenWebUIClient:
    """Client for the Open WebUI management API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the client.

        Args:
            base_url: Open WebUI base URL (e.g., https://ai.example.com)
            api_key: User's API key/Bearer token for authentication
        """
        self.base_url = (base_url or os.getenv("OPENWEBUI_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENWEBUI_API_KEY", "")
        self._provider_config_lock = asyncio.Lock()

        if not self.base_url:
            raise ValueError("Open WebUI URL required. Set OPENWEBUI_URL env var or pass base_url.")

    def _get_headers(self, api_key: Optional[str] = None) -> dict[str, str]:
        """Get request headers with authentication."""
        token = api_key or self.api_key
        headers = {
            "Content-Type": "application/json",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _path_id(value: str) -> str:
        """Encode a value used as one URL path segment."""
        return quote(value, safe="")

    @staticmethod
    def _raise_for_status(response: httpx.Response, *, request_token: str | None = None) -> None:
        """Raise an HTTP error that includes a sanitized body, never credentials."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code if error.response is not None else "unknown"
            body = error.response.text if error.response is not None else response.text
            safe_body = sanitize_error_detail(body, request_token=request_token)
            raise httpx.HTTPStatusError(
                f"HTTP {status} error response: {safe_body}",
                request=error.request,
                response=error.response,
            ) from error

    async def request(
        self,
        method: str,
        path: str,
        api_key: Optional[str] = None,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an authenticated request to Open WebUI API."""
        url = f"{self.base_url}{path}"
        headers = self._get_headers(api_key)
        client_timeout = timeout if timeout is not None else DEFAULT_HTTP_TIMEOUT_SECONDS

        async with httpx.AsyncClient(timeout=client_timeout) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                **kwargs,
            )
            self._raise_for_status(response, request_token=api_key or self.api_key)

            if response.headers.get("content-type", "").startswith("application/json"):
                payload = response.json()
                return payload if isinstance(payload, dict) else {"data": payload}
            return {"text": response.text}

    # Convenience methods
    async def get(
        self,
        path: str,
        api_key: Optional[str] = None,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict:
        return await self.request("GET", path, api_key, timeout=timeout, **kwargs)

    async def post(self, path: str, api_key: Optional[str] = None, **kwargs: Any) -> dict:
        return await self.request("POST", path, api_key, **kwargs)

    async def put(self, path: str, api_key: Optional[str] = None, **kwargs: Any) -> dict:
        return await self.request("PUT", path, api_key, **kwargs)

    async def delete(self, path: str, api_key: Optional[str] = None, **kwargs: Any) -> dict:
        return await self.request("DELETE", path, api_key, **kwargs)

    @staticmethod
    def _compact_collection(
        payload: dict[str, Any],
        fields: tuple[str, ...],
        transforms: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Keep list responses index-like; use get_* for full records."""
        transforms = transforms or {}
        collection_key = next(
            (key for key in ("data", "users", "items") if isinstance(payload.get(key), list)),
            None,
        )
        if collection_key is None:
            return payload
        compacted = []
        for item in payload[collection_key]:
            if not isinstance(item, dict):
                compacted.append(item)
                continue
            result = {}
            for key in fields:
                if key not in item:
                    continue
                value = item[key]
                if isinstance(value, str):
                    limit = 240 if key == "title" else 500
                    if len(value) > limit:
                        value = value[: limit - 1].rstrip() + "…"
                result[key] = value
            for key, transform in transforms.items():
                value = transform(item)
                if value is not None:
                    result[key] = value
            compacted.append(result)
        return {**payload, collection_key: compacted}

    @staticmethod
    def _apply_workspace_meta_fields(
        meta: dict[str, Any],
        *,
        description: Optional[str] | _UnsetType = UNSET,
        capabilities: Optional[dict[str, Any]] | _UnsetType = UNSET,
        builtin_tools: Optional[dict[str, Any]] | _UnsetType = UNSET,
        skill_ids: Optional[list[str]] | _UnsetType = UNSET,
        default_feature_ids: Optional[list[str]] | _UnsetType = UNSET,
    ) -> None:
        """Merge Open WebUI workspace meta fields using omission-preserving semantics."""
        if description is not UNSET:
            meta["description"] = None if description in (None, "") else description
        if capabilities is not UNSET:
            if capabilities is None:
                meta.pop("capabilities", None)
            else:
                existing = meta.get("capabilities")
                base = existing if isinstance(existing, dict) else {}
                meta["capabilities"] = {**base, **capabilities}
        if builtin_tools is not UNSET:
            if builtin_tools is None:
                meta.pop("builtinTools", None)
            else:
                existing = meta.get("builtinTools")
                base = existing if isinstance(existing, dict) else {}
                meta["builtinTools"] = {**base, **builtin_tools}
        if skill_ids is not UNSET:
            if skill_ids:
                meta["skillIds"] = skill_ids
            else:
                meta.pop("skillIds", None)
        if default_feature_ids is not UNSET:
            # Open WebUI UI treats missing vs [] differently for instance default fallback;
            # explicit [] must persist on the model (cf. ModelEditor defaultFeatureIds).
            meta["defaultFeatureIds"] = list(default_feature_ids or [])

    async def _assert_file_in_knowledge(
        self,
        knowledge_id: str,
        file_id: str,
        api_key: Optional[str] = None,
    ) -> None:
        """Confirm membership via paginated GET /api/v1/knowledge/{id}/files.

        Open WebUI exposes no direct file-in-KB lookup for callers; scans are capped
        by ``OPENWEBUI_KB_FILE_SCAN_MAX_PAGES`` (default 34 pages × 30 files).
        """
        max_pages = _knowledge_file_scan_max_pages()
        page = 1
        pages_fetched = 0
        next_page: int | None = None
        while pages_fetched < max_pages:
            pages_fetched += 1
            batch = await self.list_knowledge_files(
                knowledge_id,
                api_key,
                page=page,
                limit=KNOWLEDGE_FILES_LIST_PAGE_SIZE,
            )
            for item in batch.get("items", []):
                if isinstance(item, dict) and str(item.get("id")) == file_id:
                    return
            next_page = batch.get("next_page")
            if not next_page:
                break
            page = int(next_page)

        if next_page:
            cap = max_pages * KNOWLEDGE_FILES_LIST_PAGE_SIZE
            raise ValueError(_MEMBERSHIP_SCAN_LIMIT_MSG.format(cap=cap))

        raise ValueError(f"File {file_id!r} is not attached to knowledge base {knowledge_id!r}")

    async def _validate_tool_ids(self, tool_ids: list[str], api_key: Optional[str] = None) -> None:
        if not tool_ids:
            return
        tools = await self.list_tools(api_key)
        collection_key = next(
            (key for key in ("data", "items", "tools") if isinstance(tools.get(key), list)),
            None,
        )
        available = {
            str(item["id"])
            for item in (tools.get(collection_key, []) if collection_key else [])
            if isinstance(item, dict) and item.get("id")
        }
        unknown = [tool_id for tool_id in tool_ids if tool_id not in available]
        if unknown:
            raise ValueError(f"Unknown Open WebUI tool IDs: {', '.join(unknown)}")

    async def _validate_skill_ids(
        self,
        skill_ids: list[str],
        api_key: Optional[str] = None,
    ) -> None:
        """Confirm each skill exists and is readable via GET /api/v1/skills/id/{id}."""
        for skill_id in skill_ids:
            try:
                await self.get(f"/api/v1/skills/id/{self._path_id(skill_id)}", api_key)
            except httpx.HTTPStatusError as error:
                status = error.response.status_code if error.response is not None else None
                if status == 404:
                    raise ValueError(f"Unknown Open WebUI skill ID: {skill_id}") from error
                if status == 401:
                    raise ValueError(f"Inaccessible Open WebUI skill ID: {skill_id}") from error
                raise

    async def _validate_knowledge_ids(
        self,
        knowledge_ids: list[str],
        api_key: Optional[str] = None,
    ) -> None:
        """Confirm each knowledge collection exists and is readable for the caller."""
        for knowledge_id in knowledge_ids:
            try:
                await self.get(f"/api/v1/knowledge/{self._path_id(knowledge_id)}", api_key)
            except httpx.HTTPStatusError as error:
                status = error.response.status_code if error.response is not None else None
                if status == 404:
                    raise ValueError(
                        f"Unknown Open WebUI knowledge base ID: {knowledge_id}"
                    ) from error
                if status in {401, 403}:
                    raise ValueError(
                        f"Inaccessible Open WebUI knowledge base ID: {knowledge_id}"
                    ) from error
                raise

    # ==========================================================================
    # User Management
    # ==========================================================================

    async def list_users(self, api_key: Optional[str] = None) -> dict:
        """List all users (admin only), following Open WebUI's pagination."""
        first_page = await self.get("/api/v1/users/?page=1", api_key)
        users = first_page.get("users")
        total = first_page.get("total")
        if not isinstance(users, list) or not isinstance(total, int):
            return self._compact_collection(
                first_page,
                (
                    "id",
                    "email",
                    "username",
                    "name",
                    "role",
                    "last_active_at",
                    "created_at",
                    "group_ids",
                ),
            )

        all_users = list(users)
        page = 2
        while len(all_users) < total:
            next_page = await self.get(f"/api/v1/users/?page={page}", api_key)
            page_users = next_page.get("users")
            if not isinstance(page_users, list) or not page_users:
                break
            all_users.extend(page_users)
            page += 1

        first_page = {**first_page, "users": all_users}
        return self._compact_collection(
            first_page,
            (
                "id",
                "email",
                "username",
                "name",
                "role",
                "last_active_at",
                "created_at",
                "group_ids",
            ),
        )

    async def get_user(self, user_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific user."""
        return await self.get(f"/api/v1/users/{self._path_id(user_id)}", api_key)

    async def get_current_user(self, api_key: Optional[str] = None) -> dict:
        """Get the current session profile without exposing bearer tokens."""
        return sanitize_current_user(await self.get("/api/v1/auths/", api_key))

    async def update_user_role(
        self, user_id: str, role: str, api_key: Optional[str] = None
    ) -> dict:
        """Update a user's role (admin only)."""
        return await self.post(
            f"/api/v1/users/{self._path_id(user_id)}/update/role",
            api_key,
            json={"role": role},
        )

    async def delete_user(self, user_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a user (admin only)."""
        return await self.delete(f"/api/v1/users/{self._path_id(user_id)}", api_key)

    # ==========================================================================
    # Group Management
    # ==========================================================================

    async def list_groups(self, api_key: Optional[str] = None) -> dict:
        """List all groups."""
        return self._compact_collection(
            await self.get("/api/v1/groups/", api_key),
            (
                "id",
                "name",
                "description",
                "parent_group_id",
                "member_count",
                "created_at",
                "updated_at",
            ),
        )

    async def create_group(
        self,
        name: str,
        description: str = "",
        api_key: Optional[str] = None,
        *,
        parent_group_id: Optional[str] = None,
    ) -> dict:
        """Create a new group (admin only)."""
        data: dict[str, Any] = {"name": name, "description": description}
        if parent_group_id is not None:
            data["parent_group_id"] = parent_group_id
        return await self.post("/api/v1/groups/create", api_key, json=data)

    async def get_group(self, group_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific group."""
        return await self.get(f"/api/v1/groups/id/{self._path_id(group_id)}", api_key)

    async def update_group(
        self,
        group_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        parent_group_id: Optional[str] | _UnsetType = UNSET,
    ) -> dict:
        """Update a group (admin only)."""
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if parent_group_id is not UNSET:
            data["parent_group_id"] = parent_group_id
        if name is None or description is None:
            existing = await self.get_group(group_id, api_key)
            if name is None:
                data["name"] = existing["name"]
            if description is None:
                data["description"] = existing["description"]
        return await self.post(
            f"/api/v1/groups/id/{self._path_id(group_id)}/update", api_key, json=data
        )

    async def add_user_to_group(
        self, group_id: str, user_id: str, api_key: Optional[str] = None
    ) -> dict:
        """Add a user to a group (admin only)."""
        return await self.post(
            f"/api/v1/groups/id/{self._path_id(group_id)}/users/add",
            api_key,
            json={"user_ids": [user_id]},
        )

    async def remove_user_from_group(
        self, group_id: str, user_id: str, api_key: Optional[str] = None
    ) -> dict:
        """Remove a user from a group (admin only)."""
        return await self.post(
            f"/api/v1/groups/id/{self._path_id(group_id)}/users/remove",
            api_key,
            json={"user_id": user_id},
        )

    async def delete_group(self, group_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a group (admin only)."""
        return await self.delete(f"/api/v1/groups/id/{self._path_id(group_id)}/delete", api_key)

    # ==========================================================================
    # Model Management
    # ==========================================================================

    async def _list_accessible_custom_model_ids(self, api_key: Optional[str] = None) -> set[str]:
        """Return custom model IDs visible to the authenticated user."""
        first_page = await self.get("/api/v1/models/list?page=1", api_key)
        items = first_page.get("items")
        total = first_page.get("total")
        if not isinstance(items, list) or not isinstance(total, int):
            return {
                item["id"] for item in (items or []) if isinstance(item, dict) and item.get("id")
            }
        all_items = list(items)
        page = 2
        while len(all_items) < total:
            next_page = await self.get(f"/api/v1/models/list?page={page}", api_key)
            page_items = next_page.get("items")
            if not isinstance(page_items, list) or not page_items:
                break
            all_items.extend(page_items)
            page += 1
        return {item["id"] for item in all_items if isinstance(item, dict) and item.get("id")}

    async def list_models(
        self,
        kind: Literal["all", "custom", "base"] = "all",
        provider: Optional[str] = None,
        connection_id: Optional[str] = None,
        query: Optional[str] = None,
        model_id: Optional[str] = None,
        display_name: Optional[str] = None,
        status: Optional[Literal["active", "inactive"]] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """List the authenticated user's effective models with local filters.

        ``/api/models`` is intentionally the source of truth: it is scoped by
        Open WebUI to the current user's accessible models. The user-scoped
        Workspace model list is used only to classify custom records.
        """
        effective_payload = await self.get("/api/models", api_key)
        collection_key = next(
            (key for key in ("data", "items") if isinstance(effective_payload.get(key), list)),
            None,
        )
        effective_models = effective_payload.get(collection_key, []) if collection_key else []

        custom_ids = await self._list_accessible_custom_model_ids(api_key)

        models = []
        for item in effective_models:
            if not isinstance(item, dict):
                continue
            info = item.get("info") if isinstance(item.get("info"), dict) else {}
            model_id_value = item.get("id") or info.get("id")
            if not model_id_value:
                continue
            model_kind = "custom" if model_id_value in custom_ids or item.get("preset") else "base"
            if kind != "all" and model_kind != kind:
                continue
            name = item.get("name") or info.get("name") or model_id_value
            model_provider = (
                item.get("provider")
                or item.get("owned_by")
                or info.get("provider")
                or info.get("owned_by")
            )
            model_connection = item.get("connection_id")
            if model_connection is None:
                model_connection = item.get("urlIdx")
            is_active = item.get("is_active")
            if is_active is None:
                is_active = info.get("is_active", True)
            if provider and model_provider != provider:
                continue
            if connection_id is not None and str(model_connection) != str(connection_id):
                continue
            if model_id and model_id not in str(model_id_value):
                continue
            if display_name and display_name.casefold() not in str(name).casefold():
                continue
            if query:
                haystack = f"{model_id_value} {name} {model_provider or ''}".casefold()
                if query.casefold() not in haystack:
                    continue
            if status == "active" and not is_active:
                continue
            if status == "inactive" and is_active:
                continue
            compact = {
                "id": model_id_value,
                "name": name,
                "kind": model_kind,
                "is_active": bool(is_active),
            }
            if model_provider is not None:
                compact["provider"] = model_provider
            if model_connection is not None:
                compact["connection_id"] = model_connection
            models.append(compact)
        return {"data": models}

    async def update_model_access(
        self,
        model_id: str,
        access_grants: list[dict[str, Any]],
        name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Update access grants using Open WebUI's minimal model update form.

        Open WebUI accepts this as a minimal-record update; omitted model
        configuration fields may be reset or defaulted by the server.
        """
        data = {"id": model_id, "access_grants": access_grants}
        if name is not None:
            data["name"] = name
        return await self.post("/api/v1/models/model/access/update", api_key, json=data)

    async def get_model(self, model_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific custom model by ID."""
        return await self.get(f"/api/v1/models/model?id={quote(model_id, safe='')}", api_key)

    async def get_environment_info(
        self,
        api_key: Optional[str] = None,
        *,
        mcp_profile: str,
        active_operations: list[str],
    ) -> dict[str, Any]:
        """Return Open WebUI version metadata and MCP adapter capabilities (read-only)."""
        return build_environment_info(
            await self.get("/api/version", api_key),
            mcp_profile=mcp_profile,
            active_operations=active_operations,
        )

    async def validate_model_configuration(
        self,
        model_id: str,
        api_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Validate saved custom-model references and report stored settings (read-only)."""
        return await validate_model_configuration(self, model_id, api_key)

    async def test_saved_model_generation(
        self,
        model_id: str,
        prompt: str,
        *,
        max_output_tokens: int = 128,
        api_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run a bounded plain chat completion smoke test (may incur provider cost)."""
        return await run_saved_model_generation_test(
            self,
            model_id,
            prompt,
            max_output_tokens=max_output_tokens,
            api_key=api_key,
        )

    async def create_model(
        self,
        id: str,
        name: str,
        base_model_id: str,
        meta: Optional[dict] = None,
        params: Optional[dict] = None,
        access_grants: Optional[list[dict]] = None,
        knowledge_ids: Optional[list[str]] = None,
        api_key: Optional[str] = None,
        *,
        description: Optional[str] | _UnsetType = UNSET,
        capabilities: Optional[dict[str, Any]] | _UnsetType = UNSET,
        builtin_tools: Optional[dict[str, Any]] | _UnsetType = UNSET,
        skill_ids: Optional[list[str]] | _UnsetType = UNSET,
        default_feature_ids: Optional[list[str]] | _UnsetType = UNSET,
    ) -> dict:
        """Create a new model (admin only)."""
        model_meta = dict(meta or {})
        if "toolIds" in model_meta:
            await self._validate_tool_ids(model_meta.get("toolIds") or [], api_key)
        if skill_ids is not UNSET and skill_ids:
            await self._validate_skill_ids(skill_ids, api_key)
        if knowledge_ids:
            await self._validate_knowledge_ids(knowledge_ids, api_key)
        self._apply_workspace_meta_fields(
            model_meta,
            description=description,
            capabilities=capabilities,
            builtin_tools=builtin_tools,
            skill_ids=skill_ids,
            default_feature_ids=default_feature_ids,
        )
        if knowledge_ids is not None:
            model_meta["knowledge"] = [
                {"id": knowledge_id, "type": "collection"} for knowledge_id in knowledge_ids
            ]
        data = {
            "id": id,
            "name": name,
            "base_model_id": base_model_id,
            "meta": model_meta,
            "params": params or {},
            "access_grants": access_grants,
        }
        return await self.post("/api/v1/models/create", api_key, json=data)

    async def update_model(
        self,
        model_id: str,
        name: Optional[str] = None,
        meta: Optional[dict] = None,
        params: Optional[dict] = None,
        clear_params: Optional[list[str]] = None,
        access_grants: Optional[list[dict]] = None,
        base_model_id: Optional[str] = None,
        knowledge_ids: Optional[list[str]] = None,
        api_key: Optional[str] = None,
        *,
        description: Optional[str] | _UnsetType = UNSET,
        capabilities: Optional[dict[str, Any]] | _UnsetType = UNSET,
        builtin_tools: Optional[dict[str, Any]] | _UnsetType = UNSET,
        skill_ids: Optional[list[str]] | _UnsetType = UNSET,
        default_feature_ids: Optional[list[str]] | _UnsetType = UNSET,
    ) -> dict:
        """Update a model while preserving fields required by the current API."""
        existing = await self.get_model(model_id, api_key)
        model_params = {**(existing.get("params") or {}), **(params or {})}
        for key in clear_params or []:
            model_params.pop(key, None)

        model_meta = {**(existing.get("meta") or {}), **(meta or {})}
        if meta and "toolIds" in meta:
            await self._validate_tool_ids(meta.get("toolIds") or [], api_key)
        if skill_ids is not UNSET and skill_ids:
            await self._validate_skill_ids(skill_ids, api_key)
        if knowledge_ids:
            await self._validate_knowledge_ids(knowledge_ids, api_key)
        self._apply_workspace_meta_fields(
            model_meta,
            description=description,
            capabilities=capabilities,
            builtin_tools=builtin_tools,
            skill_ids=skill_ids,
            default_feature_ids=default_feature_ids,
        )
        if knowledge_ids is not None:
            existing_knowledge = model_meta.get("knowledge")
            direct_files = (
                [
                    item
                    for item in existing_knowledge
                    if isinstance(item, dict) and item.get("type") != "collection"
                ]
                if isinstance(existing_knowledge, list)
                else []
            )
            model_meta["knowledge"] = direct_files + [
                {"id": knowledge_id, "type": "collection"} for knowledge_id in knowledge_ids
            ]
        data = {
            "id": existing["id"],
            "name": name if name is not None else existing["name"],
            "base_model_id": (
                base_model_id if base_model_id is not None else existing.get("base_model_id")
            ),
            "meta": model_meta,
            "params": model_params,
            "access_grants": (
                access_grants if access_grants is not None else existing.get("access_grants")
            ),
            "is_active": existing.get("is_active", True),
        }
        if access_grants is not None:
            data["access_grants"] = access_grants
        return await self.post("/api/v1/models/model/update", api_key, json=data)

    async def _get_model_record_or_none(
        self, model_id: str, api_key: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        """Read a saved model record, returning None when a provider model is unregistered."""
        try:
            model = await self.get_model(model_id, api_key)
        except httpx.HTTPStatusError as error:
            if error.response is not None and error.response.status_code == 404:
                return None
            raise
        return model if model and model.get("id") else None

    async def _get_base_model(self, model_id: str, api_key: Optional[str] = None) -> dict[str, Any]:
        """Find an exact provider model in the admin-only base-model catalog."""
        response = await self.get("/api/models/base", api_key)
        models = response.get("data", [])
        for model in models:
            if isinstance(model, dict) and model.get("id") == model_id:
                return model
        raise ValueError(f"Provider base model {model_id!r} was not found")

    @staticmethod
    def _model_state_result(model_id: str, model: Optional[dict[str, Any]], changed: bool) -> dict:
        meta = (model or {}).get("meta") or {}
        return {
            "id": model_id,
            "name": (model or {}).get("name"),
            "visible": not bool(meta.get("hidden", False)),
            "enabled": bool((model or {}).get("is_active", True)),
            "changed": changed,
        }

    async def _save_model_state(
        self,
        model_id: str,
        record: dict[str, Any],
        *,
        create: bool,
        api_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Write the full Open WebUI model form so unrelated fields survive."""
        data = {
            "id": record.get("id", model_id),
            "name": record.get("name", model_id),
            "base_model_id": record.get("base_model_id"),
            "meta": record.get("meta") or {},
            "params": record.get("params") or {},
            "access_grants": record.get("access_grants") or [],
            "is_active": record.get("is_active", True),
        }
        endpoint = "/api/v1/models/create" if create else "/api/v1/models/model/update"
        return await self.post(endpoint, api_key, json=data)

    async def set_model_visibility(
        self, model_id: str, visible: bool, api_key: Optional[str] = None
    ) -> dict:
        """Set a model's selector visibility, creating a base-model override if needed."""
        record = await self._get_model_record_or_none(model_id, api_key)
        if record is None:
            if visible:
                base_model = await self._get_base_model(model_id, api_key)
                return self._model_state_result(
                    model_id,
                    {"name": base_model.get("name", model_id), "is_active": True},
                    changed=False,
                )
            base_model = await self._get_base_model(model_id, api_key)
            record = {
                "id": model_id,
                "name": base_model.get("name", model_id),
                "base_model_id": None,
                "meta": {"hidden": not visible},
                "params": {},
                "access_grants": [],
                "is_active": True,
            }
            create = True
        else:
            current_visible = not bool((record.get("meta") or {}).get("hidden", False))
            if current_visible == visible:
                return self._model_state_result(model_id, record, changed=False)
            record = dict(record)
            record["meta"] = {**(record.get("meta") or {}), "hidden": not visible}
            create = False

        await self._save_model_state(model_id, record, create=create, api_key=api_key)
        # Return only state fields, never prompts, parameters, or access-grant details.
        state = self._model_state_result(model_id, record, changed=True)
        state["visible"] = visible
        state["enabled"] = bool(record.get("is_active", True))
        return state

    async def set_model_enabled(
        self, model_id: str, enabled: bool, api_key: Optional[str] = None
    ) -> dict:
        """Set whether a model can be used, creating a base-model override if needed."""
        record = await self._get_model_record_or_none(model_id, api_key)
        if record is None:
            if enabled:
                base_model = await self._get_base_model(model_id, api_key)
                return self._model_state_result(
                    model_id,
                    {"name": base_model.get("name", model_id), "is_active": True},
                    changed=False,
                )
            base_model = await self._get_base_model(model_id, api_key)
            record = {
                "id": model_id,
                "name": base_model.get("name", model_id),
                "base_model_id": None,
                "meta": {},
                "params": {},
                "access_grants": [],
                "is_active": False,
            }
            create = True
        else:
            current_enabled = bool(record.get("is_active", True))
            if current_enabled == enabled:
                return self._model_state_result(model_id, record, changed=False)
            record = dict(record)
            record["is_active"] = enabled
            create = False

        await self._save_model_state(model_id, record, create=create, api_key=api_key)
        state = self._model_state_result(model_id, record, changed=True)
        state["visible"] = not bool((record.get("meta") or {}).get("hidden", False))
        state["enabled"] = enabled
        return state

    async def delete_model(self, model_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a model (admin only)."""
        return await self.post("/api/v1/models/model/delete", api_key, json={"id": model_id})

    # ==========================================================================
    # Knowledge Base Management
    # ==========================================================================

    async def list_knowledge(self, api_key: Optional[str] = None) -> dict:
        """List all knowledge bases, following Open WebUI's 30-item pages."""

        def access_grant_count(item: dict[str, Any]) -> Optional[int]:
            grants = item.get("access_grants", [])
            return len(grants) if isinstance(grants, list) else None

        first_page = await self.get("/api/v1/knowledge/?page=1", api_key)
        items = first_page.get("items")
        total = first_page.get("total")
        if (
            not isinstance(items, list)
            or not isinstance(total, int)
            or isinstance(total, bool)
            or total < 0
        ):
            payload = first_page
        else:
            all_items = list(items)
            page_count = (total + KNOWLEDGE_PAGE_SIZE - 1) // KNOWLEDGE_PAGE_SIZE
            page = 2
            while len(all_items) < total and page <= page_count:
                next_page = await self.get(f"/api/v1/knowledge/?page={page}", api_key)
                next_items = next_page.get("items")
                if not isinstance(next_items, list) or not next_items:
                    break
                all_items.extend(next_items)
                page += 1
            payload = {**first_page, "items": all_items}

        return self._compact_collection(
            payload,
            ("id", "name", "description", "file_count", "write_access", "created_at", "updated_at"),
            {"access_grant_count": access_grant_count},
        )

    async def get_knowledge(self, knowledge_id: str, api_key: Optional[str] = None) -> dict:
        """Get sanitized knowledge-base metadata (no embedded file content)."""
        payload = await self.get(f"/api/v1/knowledge/{self._path_id(knowledge_id)}", api_key)
        return project_knowledge_detail(payload if isinstance(payload, dict) else {})

    async def list_knowledge_files(
        self,
        knowledge_id: str,
        api_key: Optional[str] = None,
        *,
        page: int = 1,
        limit: int = KNOWLEDGE_FILES_LIST_PAGE_SIZE,
    ) -> dict[str, Any]:
        """List files in a knowledge base (metadata only, bounded pagination).

        ``limit`` is clamped to ``KNOWLEDGE_FILES_LIST_PAGE_SIZE`` (30) before
        ``next_page`` is derived; no role- or response-based page size inference.
        """
        page = max(1, page)
        limit = max(1, min(limit, KNOWLEDGE_FILES_LIST_PAGE_SIZE))
        payload = await self.get(
            f"/api/v1/knowledge/{self._path_id(knowledge_id)}/files",
            api_key,
            params={"page": page, "limit": limit, "include_content": False},
        )
        projected = project_knowledge_file_list_response(
            payload if isinstance(payload, dict) else {}
        )
        total = int(projected.get("total") or 0)
        projected["page"] = page
        projected["limit"] = limit
        projected["next_page"] = page + 1 if total > page * limit else None
        return projected

    async def get_file_processing_status(
        self,
        file_id: str,
        api_key: Optional[str] = None,
        *,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        """Read processing status via GET /api/v1/files/{id}/process/status (non-streaming)."""
        payload = await self.get(
            f"/api/v1/files/{self._path_id(file_id)}/process/status",
            api_key,
            params={"stream": False},
            timeout=request_timeout,
        )
        if isinstance(payload, dict) and isinstance(payload.get("status"), str):
            return {"file_id": file_id, "status": payload["status"]}
        return {"file_id": file_id, "status": "unknown"}

    async def wait_for_file_processing(
        self,
        file_id: str,
        api_key: Optional[str] = None,
        *,
        timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        """Poll file processing status until completed, failed, or timeout (bounded)."""
        timeout_seconds = max(1, min(timeout_seconds, FILE_PROCESSING_WAIT_MAX_SECONDS))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        last_status = "pending"
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            request_timeout = max(0.1, min(remaining, DEFAULT_HTTP_TIMEOUT_SECONDS))
            payload = await self.get_file_processing_status(
                file_id,
                api_key,
                request_timeout=request_timeout,
            )
            last_status = str(payload.get("status") or "pending")
            if last_status in {"completed", "failed"}:
                return {
                    "file_id": file_id,
                    "status": last_status,
                    "timed_out": False,
                }
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(FILE_PROCESSING_POLL_INTERVAL_SECONDS, remaining))
        terminal = last_status if last_status in {"completed", "failed"} else "pending"
        return {
            "file_id": file_id,
            "status": terminal,
            "timed_out": True,
        }

    async def create_knowledge(
        self,
        name: str,
        description: str = "",
        api_key: Optional[str] = None,
    ) -> dict:
        """Create a new knowledge base."""
        return await self.post(
            "/api/v1/knowledge/create",
            api_key,
            json={"name": name, "description": description},
        )

    async def update_knowledge(
        self,
        knowledge_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Update a knowledge base."""
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        return await self.post(
            f"/api/v1/knowledge/{self._path_id(knowledge_id)}/update", api_key, json=data
        )

    async def update_knowledge_access(
        self,
        knowledge_id: str,
        access_grants: list[dict[str, Any]],
        api_key: Optional[str] = None,
    ) -> dict:
        """Update knowledge-base access grants using Open WebUI's access form."""
        return await self.post(
            f"/api/v1/knowledge/{self._path_id(knowledge_id)}/access/update",
            api_key,
            json={"access_grants": access_grants},
        )

    async def delete_knowledge(self, knowledge_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a knowledge base."""
        return await self.delete(f"/api/v1/knowledge/{self._path_id(knowledge_id)}/delete", api_key)

    # ==========================================================================
    # File Management
    # ==========================================================================

    async def list_files(self, api_key: Optional[str] = None) -> dict:
        """List all files as lightweight metadata, including every page."""

        def knowledge_id(item: dict[str, Any]) -> Optional[str]:
            meta = item.get("meta") or {}
            data = meta.get("data") if isinstance(meta, dict) else None
            return data.get("knowledge_id") if isinstance(data, dict) else None

        first_page = await self.get("/api/v1/files/?page=1&content=false", api_key)
        items = first_page.get("items")
        total = first_page.get("total")
        if isinstance(items, list) and isinstance(total, int):
            all_items = list(items)
            page = 2
            while len(all_items) < total:
                next_page = await self.get(f"/api/v1/files/?page={page}&content=false", api_key)
                next_items = next_page.get("items")
                if not isinstance(next_items, list) or not next_items:
                    break
                all_items.extend(next_items)
                page += 1
            payload = {**first_page, "items": all_items}
        else:
            payload = first_page

        compacted = self._compact_collection(
            payload,
            ("id", "filename", "user_id", "hash", "created_at", "updated_at"),
            {"knowledge_id": knowledge_id},
        )
        return project_file_detail_response(compacted)

    async def upload_file(
        self,
        file_path: str,
        knowledge_id: Optional[str] = None,
        process: bool = True,
        process_in_background: bool = True,
        api_key: Optional[str] = None,
    ) -> dict:
        """Upload a local file, optionally linking it to a knowledge base."""
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            raise ValueError("file_path must be an absolute path")
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        metadata = {"knowledge_id": knowledge_id} if knowledge_id else {}
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        headers = {"Accept": "application/json"}
        token = api_key or self.api_key
        if token:
            headers["Authorization"] = f"Bearer {token}"

        params = {
            "process": str(process).lower(),
            "process_in_background": str(process_in_background).lower(),
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            with path.open("rb") as file_handle:
                response = await client.post(
                    f"{self.base_url}/api/v1/files/",
                    headers=headers,
                    params=params,
                    data={"metadata": json.dumps(metadata)},
                    files={"file": (path.name, file_handle, content_type)},
                )
            self._raise_for_status(response, request_token=api_key or self.api_key)
            payload = response.json()
            result = payload if isinstance(payload, dict) else {"data": payload}
            return strip_file_content_fields(result)

    async def search_files(
        self,
        filename: str,
        api_key: Optional[str] = None,
        *,
        skip: int = 0,
        limit: int = FILE_SEARCH_DEFAULT_LIMIT,
        knowledge_id: Optional[str] = None,
    ) -> dict:
        """Search files by filename (metadata only; never returns extracted content)."""
        if skip < 0:
            raise ValueError("skip must be zero or greater")
        if limit < 1 or limit > FILE_SEARCH_MAX_LIMIT:
            raise ValueError(f"limit must be between 1 and {FILE_SEARCH_MAX_LIMIT}")
        if knowledge_id:
            return await self._search_files_in_knowledge(
                knowledge_id,
                filename,
                api_key,
                skip=skip,
                limit=limit,
            )
        payload = await self.get(
            "/api/v1/files/search",
            api_key,
            params={
                "filename": filename,
                "content": False,
                "skip": skip,
                "limit": limit,
            },
        )
        return compact_file_search_collection(payload)

    async def _search_files_in_knowledge(
        self,
        knowledge_id: str,
        filename_pattern: str,
        api_key: Optional[str],
        *,
        skip: int,
        limit: int,
    ) -> dict[str, Any]:
        """Match filenames within one knowledge base via GET /api/v1/knowledge/{id}/files."""
        import fnmatch

        pattern = filename_pattern.casefold()
        matches: list[dict[str, Any]] = []
        page = 1
        pages_fetched = 0
        next_page: int | None = None
        while pages_fetched < _knowledge_file_scan_max_pages():
            pages_fetched += 1
            batch = await self.list_knowledge_files(
                knowledge_id,
                api_key,
                page=page,
                limit=KNOWLEDGE_FILES_LIST_PAGE_SIZE,
            )
            for item in batch.get("items", []):
                if not isinstance(item, dict):
                    continue
                name = item.get("filename")
                if isinstance(name, str) and fnmatch.fnmatch(name.casefold(), pattern):
                    matches.append(item)
            next_page = batch.get("next_page")
            if not next_page:
                break
            page = int(next_page)
        projected = compact_file_search_collection({"data": matches[skip : skip + limit]})
        if next_page:
            projected["scan_limit_reached"] = True
            projected["pages_scanned"] = pages_fetched
        return projected

    async def get_file(self, file_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific file's metadata."""
        payload = await self.get(f"/api/v1/files/{self._path_id(file_id)}", api_key)
        return project_file_detail_response(
            payload if isinstance(payload, dict) else {"data": payload}
        )

    async def get_file_content(
        self,
        file_id: str,
        api_key: Optional[str] = None,
        *,
        knowledge_id: Optional[str] = None,
        start_line: int = 1,
        max_lines: int = 200,
    ) -> dict:
        """Get a bounded slice of extracted text from GET /api/v1/files/{id}/data/content."""
        if knowledge_id:
            await self._assert_file_in_knowledge(knowledge_id, file_id, api_key)
        payload = await self.get(f"/api/v1/files/{self._path_id(file_id)}/data/content", api_key)
        return project_bounded_file_content(payload, start_line=start_line, max_lines=max_lines)

    async def _upload_file_bytes(
        self,
        filename: str,
        data: bytes,
        content_type: str,
        api_key: Optional[str] = None,
        knowledge_id: Optional[str] = None,
        *,
        process: bool = True,
        process_in_background: bool = True,
    ) -> dict:
        """Upload file bytes with Open WebUI's default background ingestion.

        Open WebUI reads ``process`` and ``process_in_background`` from query params
        (``routers/files.py``), not multipart form fields; only ``metadata`` is posted
        as form data alongside the file. The HTTP response returns the file record while
        ingestion may continue asynchronously when ``process_in_background`` is true.
        """
        url = f"{self.base_url}/api/v1/files/"
        token = api_key or self.api_key
        headers = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        metadata = {"knowledge_id": knowledge_id} if knowledge_id else {}
        files = {"file": (filename, data, content_type)}
        params = {
            "process": str(process).lower(),
            "process_in_background": str(process_in_background).lower(),
        }
        form = {"metadata": json.dumps(metadata)}
        upload_timeout = (
            FILE_UPLOAD_BACKGROUND_HTTP_TIMEOUT_SECONDS
            if process_in_background
            else FILE_UPLOAD_SYNC_HTTP_TIMEOUT_SECONDS
        )
        async with httpx.AsyncClient(timeout=upload_timeout) as client:
            response = await client.post(
                url,
                headers=headers,
                params=params,
                data=form,
                files=files,
            )
            self._raise_for_status(response, request_token=api_key or self.api_key)
            payload = response.json()
            result = payload if isinstance(payload, dict) else {"data": payload}
            return strip_file_content_fields(result)

    async def upload_text_file(
        self, filename: str, content: str, api_key: Optional[str] = None
    ) -> dict:
        """Upload a UTF-8 text file with Open WebUI background processing (default)."""
        return await self._upload_file_bytes(
            filename, content.encode("utf-8"), "text/markdown", api_key
        )

    async def add_knowledge_file(
        self,
        *,
        filename: Optional[str] = None,
        content: Optional[str] = None,
        content_base64: Optional[str] = None,
        file_id: Optional[str] = None,
        knowledge_id: Optional[str] = None,
        content_type: Optional[str] = None,
        wait_for_processing: bool = False,
        processing_wait_timeout_seconds: int = 30,
        api_key: Optional[str] = None,
    ) -> dict:
        """Upload or attach a file to Open WebUI, optionally linking it to a knowledge base.

        Uploads enqueue ingestion in the background (Open WebUI default). When
        ``wait_for_processing`` is set, poll status for up to ``processing_wait_timeout_seconds``.
        """
        sources = sum(1 for value in (content, content_base64, file_id) if value is not None)
        if sources != 1:
            raise ValueError("Provide exactly one of content, content_base64, or file_id")

        if file_id:
            if not knowledge_id:
                raise ValueError("knowledge_id is required when using file_id")
            return {
                "mode": "attach",
                "file_id": file_id,
                "knowledge": await self.add_file_to_knowledge(knowledge_id, file_id, api_key),
            }

        if not filename or not filename.strip():
            raise ValueError("filename is required when uploading content")

        name = filename.strip()
        if content is not None:
            if not content.strip():
                raise ValueError("content must not be empty")
            data = content.encode("utf-8")
            resolved_type = content_type or mimetypes.guess_type(name)[0] or "text/markdown"
        else:
            try:
                data = base64.b64decode(content_base64, validate=True)
            except Exception as exc:
                raise ValueError("content_base64 must be valid base64") from exc
            if not data:
                raise ValueError("content_base64 must not be empty")
            resolved_type = (
                content_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
            )

        if len(data) > 10 * 1024 * 1024:
            raise ValueError("uploaded content must be 10 MiB or smaller")

        uploaded = await self._upload_file_bytes(
            name,
            data,
            resolved_type,
            api_key,
            knowledge_id=knowledge_id,
        )
        uploaded_id = uploaded.get("id") or uploaded.get("file_id")
        if not uploaded_id:
            raise RuntimeError("Open WebUI did not return an uploaded file ID")

        result: dict[str, Any] = {
            "mode": "upload",
            "file": uploaded,
            "file_id": uploaded_id,
        }
        if knowledge_id:
            result["knowledge_id"] = knowledge_id
        if wait_for_processing:
            result["processing"] = await self.wait_for_file_processing(
                str(uploaded_id),
                api_key,
                timeout_seconds=processing_wait_timeout_seconds,
            )
        return result

    async def add_file_to_knowledge(
        self, knowledge_id: str, file_id: str, api_key: Optional[str] = None
    ) -> dict:
        """Attach an already-uploaded file to a knowledge base."""
        return await self.post(
            f"/api/v1/knowledge/{self._path_id(knowledge_id)}/file/add",
            api_key,
            json={"file_id": file_id},
        )

    async def update_file_content(
        self, file_id: str, content: str, api_key: Optional[str] = None
    ) -> dict:
        """Update the extracted content of a file."""
        return await self.post(
            f"/api/v1/files/{self._path_id(file_id)}/data/content/update",
            api_key,
            json={"content": content},
        )

    async def delete_file(self, file_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a file."""
        return await self.delete(f"/api/v1/files/{self._path_id(file_id)}", api_key)

    async def delete_all_files(self, api_key: Optional[str] = None) -> dict:
        """Delete all files (admin only)."""
        return await self.delete("/api/v1/files/all", api_key)

    # ==========================================================================
    # Prompt Management
    # ==========================================================================

    async def list_prompts(self, api_key: Optional[str] = None) -> dict:
        """List all prompts/templates."""
        return self._compact_collection(
            await self.get("/api/v1/prompts/", api_key),
            ("id", "command", "name", "tags", "is_active", "created_at", "updated_at"),
        )

    async def create_prompt(
        self,
        command: str,
        name: str,
        content: str,
        api_key: Optional[str] = None,
    ) -> dict:
        """Create a new prompt template."""
        return await self.post(
            "/api/v1/prompts/create",
            api_key,
            json={"command": command, "name": name, "content": content},
        )

    async def get_prompt(self, prompt_id: str, api_key: Optional[str] = None) -> dict:
        """Get a prompt by its stable Open WebUI ID."""
        return await self.get(f"/api/v1/prompts/id/{quote(prompt_id, safe='')}", api_key)

    async def update_prompt(
        self,
        prompt_id: str,
        command: Optional[str] = None,
        name: Optional[str] = None,
        content: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Update a prompt while preserving fields required by the current API."""
        existing = await self.get_prompt(prompt_id, api_key)
        data = {
            "command": command if command is not None else existing["command"],
            "name": name if name is not None else existing["name"],
            "content": content if content is not None else existing["content"],
            "data": existing.get("data"),
            "meta": existing.get("meta"),
            "tags": existing.get("tags"),
            "access_grants": existing.get("access_grants"),
        }
        return await self.post(
            f"/api/v1/prompts/id/{quote(prompt_id, safe='')}/update", api_key, json=data
        )

    async def delete_prompt(self, prompt_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a prompt template."""
        return await self.delete(f"/api/v1/prompts/id/{quote(prompt_id, safe='')}/delete", api_key)

    # ==========================================================================
    # Memory Management
    # ==========================================================================

    async def list_memories(self, api_key: Optional[str] = None) -> dict:
        """List all user memories."""
        return await self.get("/api/v1/memories/", api_key)

    async def add_memory(self, content: str, api_key: Optional[str] = None) -> dict:
        """Add a new memory."""
        return await self.post("/api/v1/memories/add", api_key, json={"content": content})

    async def query_memories(self, content: str, k: int = 5, api_key: Optional[str] = None) -> dict:
        """Query memories using semantic search."""
        return await self.post("/api/v1/memories/query", api_key, json={"content": content, "k": k})

    async def update_memory(
        self, memory_id: str, content: str, api_key: Optional[str] = None
    ) -> dict:
        """Update a memory."""
        return await self.post(
            f"/api/v1/memories/{self._path_id(memory_id)}/update",
            api_key,
            json={"content": content},
        )

    async def delete_memory(self, memory_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a memory."""
        return await self.delete(f"/api/v1/memories/{self._path_id(memory_id)}", api_key)

    async def delete_all_memories(self, api_key: Optional[str] = None) -> dict:
        """Delete all user memories."""
        return await self.delete("/api/v1/memories/delete/user", api_key)

    async def reset_memories(self, api_key: Optional[str] = None) -> dict:
        """Reset memory vector database (re-embed all memories)."""
        return await self.post("/api/v1/memories/reset", api_key)

    # ==========================================================================
    # Chat Management
    # ==========================================================================

    async def list_chats(self, api_key: Optional[str] = None) -> dict:
        """List user's chats."""
        return self._compact_collection(
            await self.get("/api/v1/chats/", api_key),
            ("id", "title", "updated_at", "created_at", "last_read_at", "active"),
        )

    async def get_chat(self, chat_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific chat."""
        return await self.get(f"/api/v1/chats/{self._path_id(chat_id)}", api_key)

    async def delete_chat(self, chat_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a chat."""
        return await self.delete(f"/api/v1/chats/{self._path_id(chat_id)}", api_key)

    async def delete_all_chats(self, api_key: Optional[str] = None) -> dict:
        """Delete all user's chats."""
        return await self.delete("/api/v1/chats/", api_key)

    async def archive_chat(self, chat_id: str, api_key: Optional[str] = None) -> dict:
        """Archive a chat."""
        return await self.post(f"/api/v1/chats/{self._path_id(chat_id)}/archive", api_key)

    async def share_chat(self, chat_id: str, api_key: Optional[str] = None) -> dict:
        """Share a chat (make public)."""
        return await self.post(f"/api/v1/chats/{self._path_id(chat_id)}/share", api_key)

    async def clone_chat(self, chat_id: str, api_key: Optional[str] = None) -> dict:
        """Clone a shared chat."""
        return await self.post(f"/api/v1/chats/{self._path_id(chat_id)}/clone", api_key)

    # ==========================================================================
    # Folder Management
    # ==========================================================================

    async def list_folders(self, api_key: Optional[str] = None) -> dict:
        """List all folders."""
        return await self.get("/api/v1/folders/", api_key)

    async def create_folder(
        self,
        name: str,
        system_prompt: Optional[str] = None,
        knowledge_ids: Optional[list[str]] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Create a folder/project with optional system prompt and collections."""
        payload: dict[str, Any] = {"name": name}
        if system_prompt is not None or knowledge_ids is not None:
            payload["data"] = {
                "system_prompt": system_prompt or "",
                "files": [
                    {"id": knowledge_id, "type": "collection"}
                    for knowledge_id in (knowledge_ids or [])
                ],
            }
        return await self.post("/api/v1/folders/", api_key, json=payload)

    async def get_folder(self, folder_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific folder."""
        return await self.get(f"/api/v1/folders/{self._path_id(folder_id)}", api_key)

    async def update_folder(
        self,
        folder_id: str,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        knowledge_ids: Optional[list[str]] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Update folder/project configuration while preserving other folder data."""
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if system_prompt is not None or knowledge_ids is not None:
            existing = await self.get_folder(folder_id, api_key)
            folder_data = dict(existing.get("data") or {})
            if system_prompt is not None:
                folder_data["system_prompt"] = system_prompt
            if knowledge_ids is not None:
                existing_files = folder_data.get("files")
                preserved_items = (
                    [
                        item
                        for item in existing_files
                        if isinstance(item, dict) and item.get("type") != "collection"
                    ]
                    if isinstance(existing_files, list)
                    else []
                )
                folder_data["files"] = preserved_items + [
                    {"id": knowledge_id, "type": "collection"} for knowledge_id in knowledge_ids
                ]
            data["data"] = folder_data
        if not data:
            raise ValueError("At least one folder field must be provided")
        return await self.post(
            f"/api/v1/folders/{self._path_id(folder_id)}/update", api_key, json=data
        )

    async def delete_folder(self, folder_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a folder."""
        return await self.delete(f"/api/v1/folders/{self._path_id(folder_id)}", api_key)

    # ==========================================================================
    # Tool Management
    # ==========================================================================

    async def list_tools(self, api_key: Optional[str] = None) -> dict:
        """List all tools."""
        return self._compact_collection(
            await self.get("/api/v1/tools/", api_key),
            ("id", "name", "updated_at", "created_at"),
            {"description": lambda item: (item.get("meta") or {}).get("description")},
        )

    async def get_tool(self, tool_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific tool."""
        return await self.get(f"/api/v1/tools/id/{self._path_id(tool_id)}", api_key)

    async def create_tool(
        self,
        id: str,
        name: str,
        content: str,
        meta: Optional[dict] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Create a new tool."""
        data = {"id": id, "name": name, "content": content, "meta": meta or {}}
        return await self.post("/api/v1/tools/create", api_key, json=data)

    async def update_tool(
        self,
        tool_id: str,
        name: Optional[str] = None,
        content: Optional[str] = None,
        meta: Optional[dict] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Update a tool using the complete form required by Open WebUI."""
        existing = await self.get_tool(tool_id, api_key)
        existing_meta = existing.get("meta")
        data = {
            "id": existing.get("id", tool_id),
            "name": name if name is not None else existing["name"],
            "content": content if content is not None else existing["content"],
            "meta": meta
            if meta is not None
            else (existing_meta if isinstance(existing_meta, dict) else {}),
        }
        return await self.post(
            f"/api/v1/tools/id/{self._path_id(tool_id)}/update", api_key, json=data
        )

    async def delete_tool(self, tool_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a tool."""
        return await self.delete(f"/api/v1/tools/id/{self._path_id(tool_id)}/delete", api_key)

    # ==========================================================================
    # Function Management
    # ==========================================================================

    async def list_functions(self, api_key: Optional[str] = None) -> dict:
        """List all functions (filters/pipes)."""
        return self._compact_collection(
            await self.get("/api/v1/functions/", api_key),
            ("id", "type", "name", "is_active", "is_global", "updated_at", "created_at"),
        )

    async def get_function(self, function_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific function."""
        return await self.get(f"/api/v1/functions/id/{self._path_id(function_id)}", api_key)

    async def create_function(
        self,
        id: str,
        name: str,
        type: str,
        content: str,
        meta: Optional[dict] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Create a new function (filter/pipe)."""
        data = {"id": id, "name": name, "type": type, "content": content}
        if meta:
            data["meta"] = meta
        return await self.post("/api/v1/functions/create", api_key, json=data)

    async def update_function(
        self,
        function_id: str,
        name: Optional[str] = None,
        content: Optional[str] = None,
        meta: Optional[dict] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Update a function."""
        data = {}
        if name is not None:
            data["name"] = name
        if content is not None:
            data["content"] = content
        if meta is not None:
            data["meta"] = meta
        return await self.post(
            f"/api/v1/functions/id/{self._path_id(function_id)}/update", api_key, json=data
        )

    async def toggle_function(self, function_id: str, api_key: Optional[str] = None) -> dict:
        """Toggle a function's enabled state."""
        return await self.post(f"/api/v1/functions/id/{self._path_id(function_id)}/toggle", api_key)

    async def delete_function(self, function_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a function."""
        return await self.delete(
            f"/api/v1/functions/id/{self._path_id(function_id)}/delete", api_key
        )

    # ==========================================================================
    # Config/Settings (Admin)
    # ==========================================================================

    async def get_config(self, api_key: Optional[str] = None) -> dict:
        """Get system configuration with credentials and secrets redacted."""
        return project_config_export(await self.get("/api/v1/configs/export", api_key))

    async def get_default_prompt_suggestions(self, api_key: Optional[str] = None) -> dict:
        """Get instance-wide prompt suggestions from the configuration export."""
        config = await self.get_config(api_key)
        suggestions = config.get("ui.prompt_suggestions")
        if suggestions is None and isinstance(config.get("ui"), dict):
            suggestions = config["ui"].get("prompt_suggestions")
        return {"suggestions": suggestions or []}

    async def set_default_prompt_suggestions(
        self, suggestions: list[dict], api_key: Optional[str] = None
    ) -> dict:
        """Set instance-wide prompt suggestions (admin only)."""
        return await self.post(
            "/api/v1/configs/suggestions",
            api_key,
            json={"suggestions": suggestions},
        )

    async def export_config(self, api_key: Optional[str] = None) -> dict:
        """Export configuration with credentials and secrets redacted."""
        return project_config_export(await self.get("/api/v1/configs/export", api_key))

    async def import_config(self, config: dict, api_key: Optional[str] = None) -> dict:
        """Import configuration (admin only)."""
        return await self.post("/api/v1/configs/import", api_key, json={"config": config})

    async def get_banners(self, api_key: Optional[str] = None) -> dict:
        """Get system banners."""
        return await self.get("/api/v1/configs/banners", api_key)

    async def set_banners(self, banners: list, api_key: Optional[str] = None) -> dict:
        """Set system banners (admin only)."""
        return await self.post("/api/v1/configs/banners", api_key, json={"banners": banners})

    async def get_models_config(self, api_key: Optional[str] = None) -> dict:
        """Get default models configuration (admin only, allowlisted fields only)."""
        return project_models_config(await self.get("/api/v1/configs/models", api_key))

    async def _get_connections_config(self, api_key: Optional[str] = None) -> dict[str, Any]:
        """Read provider settings from config export; callers must redact secrets."""
        config = await self.get("/api/v1/configs/export", api_key)
        if not isinstance(config.get("openai.api_base_urls"), list):
            raise ValueError("Open WebUI returned an invalid provider connections configuration")
        return config

    @staticmethod
    def _connection_projection(config: dict[str, Any]) -> dict[str, Any]:
        """Return connector metadata without API keys or unreviewed configuration fields."""
        base_urls = config.get("openai.api_base_urls", [])
        api_configs = config.get("openai.api_configs", {})
        if not isinstance(api_configs, dict):
            api_configs = {}
        connections = []
        for index, base_url in enumerate(base_urls):
            if not isinstance(base_url, str):
                continue
            item = api_configs.get(str(index), {})
            if not isinstance(item, dict):
                item = {}
            model_ids = item.get("model_ids", [])
            if not isinstance(model_ids, list):
                model_ids = []
            connections.append(
                {
                    "connection_index": index,
                    "base_url": sanitize_service_url(base_url),
                    "enabled": item.get("enable", True),
                    "model_ids": [value for value in model_ids if isinstance(value, str)],
                    "prefix_id": item.get("prefix_id"),
                    "provider": item.get("provider"),
                }
            )
        return {"connections": connections}

    async def get_model_connections(self, api_key: Optional[str] = None) -> dict[str, Any]:
        """List configured OpenAI-compatible connections without exposing API keys."""
        return self._connection_projection(await self._get_connections_config(api_key))

    async def set_connection_model_ids(
        self,
        connection_index: int,
        expected_base_url: str,
        expected_model_ids: list[str],
        model_ids: list[str],
        api_key: Optional[str] = None,
    ) -> dict[str, Any]:
        """Replace one connection's Model IDs while preserving all other config and secrets."""
        if connection_index < 0:
            raise ValueError("connection_index must be zero or greater")
        if not isinstance(expected_model_ids, list) or not isinstance(model_ids, list):
            raise ValueError("expected_model_ids and model_ids must be lists")

        normalized_ids = []
        for model_id in model_ids:
            if not isinstance(model_id, str) or not model_id.strip():
                raise ValueError("model_ids entries must be non-empty strings")
            normalized_ids.append(model_id.strip())
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("model_ids cannot contain duplicate entries")

        # Serialize updates through this MCP client. Open WebUI exposes no CAS or
        # revision token, so writes from other clients can still race this sequence.
        async with self._provider_config_lock:
            config = await self._get_connections_config(api_key)
            base_urls = config["openai.api_base_urls"]
            if connection_index >= len(base_urls):
                raise ValueError(f"Provider connection index {connection_index} was not found")
            actual_base_url = base_urls[connection_index]
            if actual_base_url != expected_base_url:
                raise ValueError(
                    "Provider connection URL changed; re-read configuration before updating"
                )

            existing_configs = config.get("openai.api_configs", {})
            if not isinstance(existing_configs, dict):
                raise ValueError("Open WebUI returned invalid per-connection settings")
            api_configs = dict(existing_configs)
            key = str(connection_index)
            current = api_configs.get(key, {})
            if not isinstance(current, dict):
                raise ValueError("Open WebUI returned invalid per-connection settings")
            current_ids = current.get("model_ids", [])
            if not isinstance(current_ids, list):
                current_ids = []
            current_ids = [value for value in current_ids if isinstance(value, str)]
            if current_ids != expected_model_ids:
                raise ValueError(
                    "Provider Model IDs changed; re-read configuration before updating"
                )
            if current_ids == normalized_ids:
                return {**self._connection_projection(config), "changed": False}

            api_configs[key] = {**current, "model_ids": normalized_ids}
            try:
                await self.post(
                    "/api/v1/configs/import",
                    api_key,
                    json={"config": {"openai.api_configs": api_configs}},
                )
            except httpx.HTTPStatusError as error:
                status = error.response.status_code if error.response is not None else "unknown"
                # The upstream may echo submitted configuration on failure; never surface it.
                raise RuntimeError(
                    f"Open WebUI rejected the provider Model IDs update (HTTP {status})"
                ) from None

            verified = await self._get_connections_config(api_key)
            verified_configs = verified.get("openai.api_configs", {})
            verified_entry = (
                verified_configs.get(key, {}) if isinstance(verified_configs, dict) else {}
            )
            verified_ids = (
                verified_entry.get("model_ids", []) if isinstance(verified_entry, dict) else []
            )
            if verified_ids != normalized_ids:
                raise RuntimeError("Open WebUI did not persist the requested provider Model IDs")
            return {**self._connection_projection(verified), "changed": True}

    async def set_models_config(
        self,
        default_models: Optional[str] = None,
        model_order: Optional[list] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Set default models configuration (admin only)."""
        data = {}
        if default_models is not None:
            data["DEFAULT_MODELS"] = default_models
        if model_order is not None:
            data["MODEL_ORDER_LIST"] = model_order
        return await self.post("/api/v1/configs/models", api_key, json=data)

    async def _fetch_tool_servers_config(self, api_key: Optional[str] = None) -> dict[str, Any]:
        return await self.get("/api/v1/configs/tool_servers", api_key)

    async def get_tool_servers(self, api_key: Optional[str] = None) -> dict:
        """Get tool server connection metadata without auth material."""
        return sanitize_tool_servers_config(await self._fetch_tool_servers_config(api_key))

    async def set_tool_servers(self, connections: list, api_key: Optional[str] = None) -> dict:
        """Set tool server connections (admin only)."""
        return await self.post(
            "/api/v1/configs/tool_servers",
            api_key,
            json={"TOOL_SERVER_CONNECTIONS": connections},
        )

    async def update_tool_server_config(
        self,
        server_id: str,
        function_name_filter_list: list[str],
        description: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Update one tool server's filter and optional description (admin only)."""
        config = await self._fetch_tool_servers_config(api_key)
        connections = config.get("TOOL_SERVER_CONNECTIONS", [])
        matching = [
            connection
            for connection in connections
            if (connection.get("info") or {}).get("id") == server_id
        ]
        if len(matching) != 1:
            raise ValueError(f"Expected exactly one tool server with id {server_id!r}")

        connection = matching[0]
        connection.setdefault("config", {})["function_name_filter_list"] = function_name_filter_list
        if description is not None:
            connection.setdefault("info", {})["description"] = description

        result = await self.set_tool_servers(connections, api_key)
        return sanitize_tool_servers_config(result)

    # ==========================================================================
    # Notes Management
    # ==========================================================================

    async def list_notes(self, api_key: Optional[str] = None) -> dict:
        """List all notes."""
        return self._compact_collection(
            await self.get("/api/v1/notes/", api_key),
            ("id", "title", "is_pinned", "updated_at", "created_at"),
        )

    async def create_note(
        self,
        title: str,
        content: str,
        api_key: Optional[str] = None,
    ) -> dict:
        """Create a new note."""
        return await self.post(
            "/api/v1/notes/create",
            api_key,
            json={"title": title, "data": {"content": {"md": content}}},
        )

    async def get_note(self, note_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific note."""
        return await self.get(f"/api/v1/notes/{self._path_id(note_id)}", api_key)

    async def update_note(
        self,
        note_id: str,
        title: Optional[str] = None,
        content: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Update a note."""
        data = {}
        if title is not None:
            data["title"] = title
        if content is not None:
            data["data"] = {"content": {"md": content}}
        return await self.post(f"/api/v1/notes/{self._path_id(note_id)}/update", api_key, json=data)

    async def delete_note(self, note_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a note."""
        return await self.delete(f"/api/v1/notes/{self._path_id(note_id)}/delete", api_key)

    # ==========================================================================
    # Channels (Team Chat) Management
    # ==========================================================================

    async def list_channels(self, api_key: Optional[str] = None) -> dict:
        """List channels accessible to the user."""
        return self._compact_collection(
            await self.get("/api/v1/channels/", api_key),
            (
                "id",
                "type",
                "name",
                "description",
                "is_private",
                "last_message_at",
                "unread_count",
                "created_at",
                "updated_at",
            ),
            {"member_count": lambda item: len(item.get("user_ids") or [])},
        )

    async def create_channel(
        self,
        name: str,
        description: str = "",
        api_key: Optional[str] = None,
    ) -> dict:
        """Create a new channel (admin only)."""
        return await self.post(
            "/api/v1/channels/create",
            api_key,
            json={"name": name, "description": description},
        )

    async def get_channel(self, channel_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific channel."""
        return await self.get(f"/api/v1/channels/{self._path_id(channel_id)}", api_key)

    async def update_channel(
        self,
        channel_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Update a channel (admin only)."""
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        return await self.post(
            f"/api/v1/channels/{self._path_id(channel_id)}/update", api_key, json=data
        )

    async def delete_channel(self, channel_id: str, api_key: Optional[str] = None) -> dict:
        """Delete a channel (admin only)."""
        return await self.delete(f"/api/v1/channels/{self._path_id(channel_id)}/delete", api_key)

    async def get_channel_messages(
        self,
        channel_id: str,
        skip: int = 0,
        limit: int = 50,
        api_key: Optional[str] = None,
    ) -> dict:
        """Get messages from a channel."""
        return await self.get(
            f"/api/v1/channels/{self._path_id(channel_id)}/messages?skip={skip}&limit={limit}",
            api_key,
        )

    async def post_channel_message(
        self,
        channel_id: str,
        content: str,
        parent_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Post a message to a channel."""
        data = {"content": content}
        if parent_id:
            data["parent_id"] = parent_id
        return await self.post(
            f"/api/v1/channels/{self._path_id(channel_id)}/messages/post",
            api_key,
            json=data,
        )

    async def delete_channel_message(
        self,
        channel_id: str,
        message_id: str,
        api_key: Optional[str] = None,
    ) -> dict:
        """Delete a message from a channel."""
        return await self.delete(
            f"/api/v1/channels/{self._path_id(channel_id)}/messages/"
            f"{self._path_id(message_id)}/delete",
            api_key,
        )
