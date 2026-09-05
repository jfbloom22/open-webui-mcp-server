"""Open WebUI API client using a locally configured management credential."""

import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote

import httpx

KNOWLEDGE_PAGE_SIZE = 30


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

    async def request(
        self,
        method: str,
        path: str,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an authenticated request to Open WebUI API."""
        url = f"{self.base_url}{path}"
        headers = self._get_headers(api_key)

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                **kwargs,
            )
            response.raise_for_status()

            if response.headers.get("content-type", "").startswith("application/json"):
                payload = response.json()
                return payload if isinstance(payload, dict) else {"data": payload}
            return {"text": response.text}

    # Convenience methods
    async def get(self, path: str, api_key: Optional[str] = None, **kwargs: Any) -> dict:
        return await self.request("GET", path, api_key, **kwargs)

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
        """Get the currently authenticated user."""
        return await self.get("/api/v1/auths/", api_key)

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
            ("id", "name", "description", "member_count", "created_at", "updated_at"),
        )

    async def create_group(
        self, name: str, description: str = "", api_key: Optional[str] = None
    ) -> dict:
        """Create a new group (admin only)."""
        return await self.post(
            "/api/v1/groups/create",
            api_key,
            json={"name": name, "description": description},
        )

    async def get_group(self, group_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific group."""
        return await self.get(f"/api/v1/groups/id/{self._path_id(group_id)}", api_key)

    async def update_group(
        self,
        group_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Update a group (admin only)."""
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
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

    async def _list_accessible_custom_model_ids(
        self, api_key: Optional[str] = None
    ) -> set[str]:
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

    async def create_model(
        self,
        id: str,
        name: str,
        base_model_id: str,
        meta: Optional[dict] = None,
        params: Optional[dict] = None,
        access_grants: Optional[list[dict]] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Create a new model (admin only)."""
        data = {
            "id": id,
            "name": name,
            "base_model_id": base_model_id,
            "meta": meta or {},
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
        access_grants: Optional[list[dict]] = None,
        base_model_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Update a model while preserving fields required by the current API."""
        existing = await self.get_model(model_id, api_key)
        data = {
            "id": existing["id"],
            "name": name if name is not None else existing["name"],
            "base_model_id": (
                base_model_id if base_model_id is not None else existing.get("base_model_id")
            ),
            "meta": {**(existing.get("meta") or {}), **(meta or {})},
            "params": {**(existing.get("params") or {}), **(params or {})},
            "access_grants": (
                access_grants if access_grants is not None else existing.get("access_grants")
            ),
            "is_active": existing.get("is_active", True),
        }
        if access_grants is not None:
            data["access_grants"] = access_grants
        return await self.post("/api/v1/models/model/update", api_key, json=data)

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
        """Get a specific knowledge base."""
        return await self.get(f"/api/v1/knowledge/{self._path_id(knowledge_id)}", api_key)

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
                next_page = await self.get(
                    f"/api/v1/files/?page={page}&content=false", api_key
                )
                next_items = next_page.get("items")
                if not isinstance(next_items, list) or not next_items:
                    break
                all_items.extend(next_items)
                page += 1
            payload = {**first_page, "items": all_items}
        else:
            payload = first_page

        return self._compact_collection(
            payload,
            ("id", "filename", "user_id", "hash", "created_at", "updated_at"),
            {"knowledge_id": knowledge_id},
        )

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
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {"data": payload}

    async def search_files(self, filename: str, api_key: Optional[str] = None) -> dict:
        """Search files by filename pattern (supports wildcards)."""
        return await self.get("/api/v1/files/search", api_key, params={"filename": filename})

    async def get_file(self, file_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific file's metadata."""
        return await self.get(f"/api/v1/files/{self._path_id(file_id)}", api_key)

    async def get_file_content(self, file_id: str, api_key: Optional[str] = None) -> dict:
        """Get extracted text content from a file."""
        return await self.get(f"/api/v1/files/{self._path_id(file_id)}/data/content", api_key)

    async def upload_text_file(
        self, filename: str, content: str, api_key: Optional[str] = None
    ) -> dict:
        """Upload a UTF-8 text file and let Open WebUI process it synchronously."""
        url = f"{self.base_url}/api/v1/files/"
        token = api_key or self.api_key
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        files = {"file": (filename, content.encode("utf-8"), "text/markdown")}
        data = {"process": "true", "process_in_background": "false"}
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, headers=headers, files=files, data=data)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {"data": payload}

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

    async def create_folder(self, name: str, api_key: Optional[str] = None) -> dict:
        """Create a new folder."""
        return await self.post("/api/v1/folders/", api_key, json={"name": name})

    async def get_folder(self, folder_id: str, api_key: Optional[str] = None) -> dict:
        """Get a specific folder."""
        return await self.get(f"/api/v1/folders/{self._path_id(folder_id)}", api_key)

    async def update_folder(self, folder_id: str, name: str, api_key: Optional[str] = None) -> dict:
        """Update a folder's name."""
        return await self.post(
            f"/api/v1/folders/{self._path_id(folder_id)}/update", api_key, json={"name": name}
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
        data = {"id": id, "name": name, "content": content}
        if meta:
            data["meta"] = meta
        return await self.post("/api/v1/tools/create", api_key, json=data)

    async def update_tool(
        self,
        tool_id: str,
        name: Optional[str] = None,
        content: Optional[str] = None,
        meta: Optional[dict] = None,
        api_key: Optional[str] = None,
    ) -> dict:
        """Update a tool."""
        data = {}
        if name is not None:
            data["name"] = name
        if content is not None:
            data["content"] = content
        if meta is not None:
            data["meta"] = meta
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
        """Get the full system configuration (admin only)."""
        return await self.get("/api/v1/configs/export", api_key)

    async def export_config(self, api_key: Optional[str] = None) -> dict:
        """Export full configuration (admin only)."""
        return await self.get("/api/v1/configs/export", api_key)

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
        """Get default models configuration (admin only)."""
        return await self.get("/api/v1/configs/models", api_key)

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

    async def get_tool_servers(self, api_key: Optional[str] = None) -> dict:
        """Get tool server connections (admin only)."""
        return await self.get("/api/v1/configs/tool_servers", api_key)

    async def set_tool_servers(self, connections: list, api_key: Optional[str] = None) -> dict:
        """Set tool server connections (admin only)."""
        return await self.post(
            "/api/v1/configs/tool_servers",
            api_key,
            json={"TOOL_SERVER_CONNECTIONS": connections},
        )

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
            json={"title": title, "content": content},
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
            data["content"] = content
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
