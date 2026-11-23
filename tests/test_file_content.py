from openwebui_mcp.security import project_bounded_file_content, project_knowledge_detail


def test_project_bounded_file_content_defaults_to_two_hundred_lines() -> None:
    lines = "\n".join(f"line {index}" for index in range(1, 301))
    result = project_bounded_file_content({"content": lines}, start_line=1, max_lines=200)

    assert result["line_count"] == 200
    assert result["start_line"] == 1
    assert result["end_line"] == 200
    assert result["total_lines"] == 300
    assert result["has_more"] is True


def test_project_bounded_file_content_respects_range_cap() -> None:
    text = "\n".join("x" for _ in range(10))
    capped = project_bounded_file_content({"content": text}, start_line=1, max_lines=600)
    assert capped["line_count"] == 10

    result = project_bounded_file_content({"content": text}, start_line=5, max_lines=500)
    assert result["line_count"] == 6
    assert result["start_line"] == 5
    assert result["end_line"] == 10


def test_project_knowledge_detail_omits_raw_grants_and_meta_secrets() -> None:
    payload = {
        "id": "kb-1",
        "name": "KB",
        "meta": {
            "source": "external",
            "read_only": True,
            "apiKey": "secret",
            "external": {"provider": "s3"},
        },
        "access_grants": [{"principal_id": "g1"}],
        "files": [{"id": "f1", "data": {"content": "x"}}],
    }

    result = project_knowledge_detail(payload)

    assert result["access_grant_count"] == 1
    assert "access_grants" not in result
    assert result["meta"] == {
        "source": "external",
        "read_only": True,
        "external": {"provider": "s3"},
    }
    assert "secret" not in str(result)
