# Changelog

## Unreleased

### Added

- Read-only `get_environment_info` (Open WebUI `GET /api/version`, MCP package
  version, `MCP_PROFILE`, and runtime-active MCP tools after profile filtering).
  Tool names come from ``advertised_mcp_tool_names`` (FastMCP 4 ``await mcp.list_tools()``).
- `list_knowledge_files` for metadata-only paginated KB file listings
  (`GET /api/v1/knowledge/{id}/files`, `include_content=false`); MCP enforces
  `limit` ≤ 30 for pagination.
- `get_file_processing_status` via `GET /api/v1/files/{id}/process/status`
  (`stream=false`).
- `get_knowledge_base` now returns sanitized KB metadata without embedded file
  payloads; use `list_knowledge_files` for file listings.
- Optional bounded processing wait on `add_knowledge_file` uploads (max 30s poll;
  background processing on upload when enabled; poll HTTP timeouts respect remaining budget).
- Byte uploads send `process` / `process_in_background` as query params per Open WebUI
  `POST /api/v1/files/` (multipart carries metadata and file only); default background
  ingestion matches Open WebUI (`process_in_background=true`).
- `get_file_content` returns bounded line windows (default 200, max 500 lines).
- `search_files` optional `knowledge_id` scopes via knowledge file listing, not global
  search; sets `scan_limit_reached` when the 34-page scan cap applies.
- Explicit admin/local `test_saved_model_generation` smoke test via bounded
  `POST /api/chat/completions` (plain user message, no chat persistence or tool
  loop; optional `version_id` observation with concurrency caveat).
- Read-only `validate_model_configuration` for the current caller: base-model
  catalog availability, tool/skill/knowledge checks, `function_calling` support
  and Open WebUI `native` default when omitted, and stored vs globally-gated
  effective fields for capabilities, builtin tools, and default features.
- Group hierarchy support: `parent_group_id` on list, create, and update group
  operations. Updates omitting `parent_group_id` preserve the current parent;
  an explicit null parent moves the group to the root level.
- Workspace model tools now accept `description`, `function_calling`,
  `capabilities`, `builtin_tools`, `skill_ids`, and `default_feature_ids` with
  omission-preserving updates and merge-by-key semantics for object fields.
- `get_file_content` accepts optional `knowledge_id` and verifies file linkage
  server-side before returning extracted text.

### Changed

- `create_model` and `update_model` validate non-empty `knowledge_ids` with
  `GET /api/v1/knowledge/{id}` before POST (unknown or inaccessible IDs fail
  before write).
- `get_current_user` omits the session `token` field entirely; `{present}` markers
  remain for configuration credential keys only.
- Replaced `add_knowledge_text` with unified `add_knowledge_file` supporting
  text uploads, base64 binary uploads, and attaching existing Open WebUI files
  by `file_id`.
- `search_files` passes `content=false`, `skip`, and `limit` (default 100, max
  1000) and returns the narrow search allowlist (`id`, `filename`,
  `knowledge_ids`, `status`, `content_type`, `size`) without raw `meta`/`data`.
- `get_system_config`, `export_config`, `get_models_config`, and `get_tool_servers`
  use allowlisted projections; credential fields are `{present}` only. HTTP errors
  omit non-JSON upstream bodies.
- `get_file_content` knowledge scoping checks membership via paginated
  `list_knowledge_files`; scan cap errors when inconclusive (configurable via
  `OPENWEBUI_KB_FILE_SCAN_MAX_PAGES`).
- File upload/list metadata responses no longer include extracted `content`
  fields unless explicitly read through `get_file_content`.

### Fixed

- Uploads that include `knowledge_id` now pass `metadata.knowledge_id` on file
  creation so Open WebUI links synchronously instead of racing a second attach
  call.

## 0.2.0 - 2026-09-05

### Added

- Added access sharing for knowledge bases through the MCP management tool.
- Added file upload support with optional knowledge-base linking.
- Added discovery of effective provider, base, and custom models.

### Fixed

- Followed pagination for users, files, and knowledge bases.
- Avoided downloading extracted file content during file listings.
- Corrected system configuration lookup to use Open WebUI's export endpoint.
- Aligned management routes with current Open WebUI APIs.
- Encoded query parameters and resource IDs safely.
- Forwarded authenticated Open WebUI session tokens correctly for scoped profiles.

