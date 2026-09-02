from unittest.mock import AsyncMock, Mock, patch

import pytest

from openwebui_mcp import main


def test_member_profile_allowlist_excludes_admin_surfaces() -> None:
    assert "update_model" in main.MEMBER_PROFILE_TOOLS
    assert "add_knowledge_text" in main.MEMBER_PROFILE_TOOLS
    assert "list_users" not in main.MEMBER_PROFILE_TOOLS
    assert "get_tool_servers" not in main.MEMBER_PROFILE_TOOLS
    assert not any(name.startswith("delete_") for name in main.MEMBER_PROFILE_TOOLS)


def test_deployed_profiles_require_forwarded_session_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_PROFILE", "member")
    token = main._current_user_token.set(None)
    try:
        with pytest.raises(RuntimeError, match="forwarded Open WebUI session token"):
            main.get_user_token()
    finally:
        main._current_user_token.reset(token)


@pytest.mark.asyncio
async def test_configure_member_profile_removes_unapproved_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_PROFILE", "member")
    registered = {"list_models": Mock(), "update_model": Mock(), "delete_model": Mock()}
    get_tools = AsyncMock(return_value=registered)
    remove_tool = Mock()
    with patch.object(main.mcp, "get_tools", get_tools), patch.object(
        main.mcp, "remove_tool", remove_tool
    ):
        await main.configure_profile()

    remove_tool.assert_called_once_with("delete_model")
