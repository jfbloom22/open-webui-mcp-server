# Changelog

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

