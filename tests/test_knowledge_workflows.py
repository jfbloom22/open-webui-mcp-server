from unittest.mock import AsyncMock

import httpx
import pytest

from openwebui_mcp.client import OpenWebUIClient
from openwebui_mcp.security import project_knowledge_detail, project_knowledge_file_list_response


@pytest.mark.asyncio
async def test_search_files_with_knowledge_id_uses_collection_listing() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock()
    client.list_knowledge_files = AsyncMock(
        return_value={
            "items": [
                {"id": "f1", "filename": "notes.md"},
                {"id": "f2", "filename": "other.txt"},
            ],
            "total": 2,
            "next_page": None,
        }
    )

    result = await client.search_files("*.md", "token", knowledge_id="kb-1")

    client.list_knowledge_files.assert_awaited()
    client.get.assert_not_called()
    assert result["data"] == [{"id": "f1", "filename": "notes.md"}]
    assert "scan_limit_reached" not in result


def test_project_knowledge_detail_omits_embedded_file_content() -> None:
    payload = {
        "id": "kb-1",
        "name": "Research",
        "description": "KB",
        "write_access": True,
        "files": [
            {
                "id": "file-1",
                "filename": "notes.md",
                "data": {"content": "secret body", "status": "completed"},
            }
        ],
    }

    result = project_knowledge_detail(payload)

    assert result["id"] == "kb-1"
    assert result["file_count"] == 1
    assert "files" not in result or result.get("files") is None
    assert result["file_listing"]["tool"] == "list_knowledge_files"
    assert "secret body" not in str(result)


def test_project_knowledge_file_list_response_strips_content() -> None:
    payload = {
        "items": [
            {
                "id": "file-1",
                "filename": "notes.md",
                "meta": {"size": 10},
                "data": {"content": "extracted", "status": "processing"},
            }
        ],
        "total": 1,
        "directories": [{"id": "dir-1", "name": "Docs", "parent_id": None, "knowledge_id": "kb-1"}],
    }

    result = project_knowledge_file_list_response(payload)

    assert result["items"] == [
        {"id": "file-1", "filename": "notes.md", "status": "processing", "size": 10}
    ]
    assert "extracted" not in str(result)
    assert result["directories"] == [
        {"id": "dir-1", "name": "Docs", "parent_id": None, "knowledge_id": "kb-1"}
    ]


@pytest.mark.asyncio
async def test_list_knowledge_files_uses_supported_endpoint_params() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        return_value={
            "items": [{"id": "file-1", "filename": "a.md", "data": {"status": "completed"}}],
            "total": 45,
        }
    )

    result = await client.list_knowledge_files("kb/slug", "token", page=2, limit=20)

    client.get.assert_awaited_once_with(
        "/api/v1/knowledge/kb%2Fslug/files",
        "token",
        params={"page": 2, "limit": 20, "include_content": False},
    )
    assert result["page"] == 2
    assert result["next_page"] == 3
    assert result["items"][0]["id"] == "file-1"
    assert "content" not in str(result["items"])


@pytest.mark.asyncio
async def test_list_knowledge_files_clamps_limit_and_computes_next_page() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        return_value={
            "items": [{"id": "file-1", "filename": "a.md"}],
            "total": 45,
        }
    )

    result = await client.list_knowledge_files("kb-1", "token", page=1, limit=100)

    client.get.assert_awaited_once_with(
        "/api/v1/knowledge/kb-1/files",
        "token",
        params={"page": 1, "limit": 30, "include_content": False},
    )
    assert result["limit"] == 30
    assert result["next_page"] == 2

    client.get = AsyncMock(
        return_value={
            "items": [{"id": "file-2", "filename": "b.md"}],
            "total": 45,
        }
    )
    result = await client.list_knowledge_files("kb-1", "token", page=2, limit=20)

    assert result["limit"] == 20
    assert result["next_page"] == 3


@pytest.mark.asyncio
async def test_list_knowledge_files_does_not_clamp_page() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(
        return_value={
            "items": [{"id": "file-1", "filename": "a.md"}],
            "total": 20_000,
        }
    )

    result = await client.list_knowledge_files("kb-1", "token", page=500, limit=30)

    client.get.assert_awaited_once_with(
        "/api/v1/knowledge/kb-1/files",
        "token",
        params={"page": 500, "limit": 30, "include_content": False},
    )
    assert result["page"] == 500
    assert result["next_page"] == 501

    client.get = AsyncMock(return_value={"items": [], "total": 20_000})
    await client.list_knowledge_files("kb-1", "token", page=501)

    assert client.get.await_args.kwargs["params"]["page"] == 501


@pytest.mark.asyncio
async def test_search_files_knowledge_scan_sets_limit_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    import openwebui_mcp.client as client_module

    monkeypatch.setattr(client_module, "KNOWLEDGE_FILE_LIST_SCAN_MAX_PAGES", 2)

    client = OpenWebUIClient(base_url="https://webui.example")

    async def fake_list(
        _knowledge_id: str,
        _token: str,
        *,
        page: int = 1,
        limit: int = 30,
    ) -> dict:
        return {
            "items": [{"id": f"f{page}", "filename": "match.md"}],
            "total": 999,
            "next_page": page + 1,
        }

    client.list_knowledge_files = fake_list

    result = await client.search_files("*.md", "token", knowledge_id="kb-1")

    assert len(result["data"]) == 2
    assert result["data"][0]["id"] == "f1"
    assert result["scan_limit_reached"] is True
    assert result["pages_scanned"] == 2


@pytest.mark.asyncio
async def test_get_file_processing_status_uses_non_stream_endpoint() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client.get = AsyncMock(return_value={"status": "processing"})

    result = await client.get_file_processing_status("file-1", "token")

    client.get.assert_awaited_once_with(
        "/api/v1/files/file-1/process/status",
        "token",
        params={"stream": False},
        timeout=None,
    )
    assert result == {"file_id": "file-1", "status": "processing"}


@pytest.mark.asyncio
async def test_get_file_processing_status_maps_404_to_http_error() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    request = httpx.Request("GET", "https://webui.example/api/v1/files/missing/process/status")
    response = httpx.Response(404, request=request, text='{"detail":"hidden"}')
    client.get = AsyncMock(
        side_effect=httpx.HTTPStatusError("404", request=request, response=response)
    )

    with pytest.raises(httpx.HTTPStatusError) as error:
        await client.get_file_processing_status("missing", "token")

    assert "hidden" not in str(error.value)


@pytest.mark.asyncio
async def test_wait_for_file_processing_passes_remaining_request_timeout() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    observed: list[float | None] = []

    async def fake_status(
        _file_id: str,
        _token: str,
        *,
        request_timeout: float | None = None,
    ) -> dict:
        observed.append(request_timeout)
        if len(observed) == 1:
            return {"file_id": _file_id, "status": "pending"}
        return {"file_id": _file_id, "status": "completed"}

    client.get_file_processing_status = fake_status

    await client.wait_for_file_processing("file-1", "token", timeout_seconds=5)

    assert observed[0] is not None
    assert 0 < observed[0] <= 5


@pytest.mark.asyncio
async def test_wait_for_file_processing_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    statuses = iter(["pending", "completed"])

    async def fake_status(_file_id: str, _token: str, **_kwargs) -> dict:
        return {"file_id": _file_id, "status": next(statuses)}

    client.get_file_processing_status = fake_status
    monkeypatch.setattr("openwebui_mcp.client.asyncio.sleep", AsyncMock())

    result = await client.wait_for_file_processing("file-1", "token", timeout_seconds=10)

    assert result == {"file_id": "file-1", "status": "completed", "timed_out": False}


@pytest.mark.asyncio
async def test_add_knowledge_file_optional_wait_after_upload() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client._upload_file_bytes = AsyncMock(return_value={"id": "file-1"})
    client.wait_for_file_processing = AsyncMock(
        return_value={"file_id": "file-1", "status": "completed", "timed_out": False}
    )

    result = await client.add_knowledge_file(
        filename="notes.md",
        content="# Hi",
        knowledge_id="kb-1",
        wait_for_processing=True,
        processing_wait_timeout_seconds=15,
        api_key="token",
    )

    client.wait_for_file_processing.assert_awaited_once_with("file-1", "token", timeout_seconds=15)
    assert result["processing"]["status"] == "completed"


@pytest.mark.asyncio
async def test_assert_file_in_knowledge_reports_scan_limit_instead_of_false_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openwebui_mcp.client as client_module

    monkeypatch.setattr(client_module, "KNOWLEDGE_FILE_LIST_SCAN_MAX_PAGES", 2)

    client = OpenWebUIClient(base_url="https://webui.example")

    async def fake_list(
        _knowledge_id: str,
        _token: str,
        *,
        page: int = 1,
        limit: int = 30,
    ) -> dict:
        return {
            "items": [{"id": f"other-{page}"}],
            "total": 999,
            "page": page,
            "limit": limit,
            "next_page": page + 1,
        }

    client.list_knowledge_files = fake_list

    with pytest.raises(ValueError, match="membership could not be fully verified"):
        await client._assert_file_in_knowledge("kb-1", "target-file", "token")


@pytest.mark.asyncio
async def test_add_knowledge_file_without_wait_still_uploads_in_background() -> None:
    client = OpenWebUIClient(base_url="https://webui.example")
    client._upload_file_bytes = AsyncMock(return_value={"id": "file-1"})

    await client.add_knowledge_file(
        filename="notes.md",
        content="# Hi",
        api_key="token",
    )

    assert "process_in_background" not in client._upload_file_bytes.await_args.kwargs
