# Open WebUI MCP Server

An MCP (Model Context Protocol) server that exposes Open WebUI's admin APIs as tools, allowing AI assistants to manage users, groups, models, knowledge bases, and more.

List tools return compact indexes containing identifiers and discovery metadata. Use the corresponding `get_*` tool when the full configuration, content, memberships, or permissions are needed.

## Features

- **User Management**: List, get, update roles, delete users
- **Group Management**: Create, update, add/remove members, delete groups
- **Model Management**: Discover user-scoped provider, base, and custom models; create custom models, update settings, and manage access grants
- **Knowledge Base Management**: Create, list, update, share, and delete knowledge bases
- **File Management**: Upload local files, optionally linking them to a knowledge base, and manage file content
- **Chat Management**: List, view, delete chats
- **Tool & Function Discovery**: List available tools and functions
- **Permission-Aware**: All operations respect the logged-in user's permissions

## Security

The normal deployment is a local stdio process. It uses `OPENWEBUI_API_KEY` from
the local environment and is intended for a single trusted user.

- Delete tools are disabled before the server advertises its MCP tools.
- Add further disabled tool names with `OPENWEBUI_DISABLED_TOOLS`, as a
  comma-separated list.
- For the optional HTTP transport, bind to loopback and set `MCP_HTTP_TOKEN`.
  HTTP requests without that bearer token are rejected and the token is never
  forwarded to Open WebUI.

## Installation

```bash
pip install openwebui-mcp-server
```

Or with uv:

```bash
uv pip install openwebui-mcp-server
```

## Configuration

Copy the example file and set the required Open WebUI management credential:

```bash
cp .env.example .env
```

## Usage

### With a local MCP client

Run the server directly with a client that supports stdio, or use the local
Podman image:

```bash
podman build -t openwebui-mcp:local .
podman run --interactive --rm --env-file .env openwebui-mcp:local
```

Each local MCP client may start its own stdio container. The server has no local
shared state, so concurrent clients can independently manage the same Open
WebUI instance.

### Optional loopback HTTP transport

HTTP is reserved for a future persistent local service. Set these values in
your local `.env` before starting it:

```bash
export MCP_TRANSPORT=http
export MCP_HTTP_HOST=127.0.0.1
export MCP_HTTP_PORT=8001
export MCP_HTTP_TOKEN=generate-a-long-random-token
```

The connecting client must send `Authorization: Bearer <MCP_HTTP_TOKEN>`. Do
not expose this endpoint beyond the local machine.

### Programmatic Usage

```python
from openwebui_mcp.client import OpenWebUIClient

client = OpenWebUIClient(
    base_url="https://your-openwebui-instance.com",
    api_key="your-api-key"
)

# List all users (admin only)
users = await client.list_users()

# Create a group
group = await client.create_group("Engineering", "Engineering team")

# Create a custom model
model = await client.create_model(
    id="my-assistant",
    name="My Assistant",
    base_model_id="gpt-4",
    meta={"system": "You are a helpful assistant."},
    params={"temperature": 0.7}
)
```

## Available Tools

The tables below describe the local runtime's broad administrative inventory.
It does not advertise delete tools. Use `OPENWEBUI_DISABLED_TOOLS` to exclude
additional tools when needed.

### User Management
| Tool | Description | Permission |
|------|-------------|------------|
| `get_current_user` | Get authenticated user's profile | Any |
| `list_users` | List all users | Admin |
| `get_user` | Get specific user details | Admin |
| `update_user_role` | Change user role | Admin |
| `delete_user` | Delete a user | Admin |

### Group Management
| Tool | Description | Permission |
|------|-------------|------------|
| `list_groups` | List all groups | Any |
| `create_group` | Create a new group | Admin |
| `get_group` | Get group details | Any |
| `update_group` | Update group name/description | Admin |
| `add_user_to_group` | Add user to group | Admin |
| `remove_user_from_group` | Remove user from group | Admin |
| `delete_group` | Delete a group | Admin |

### Model Management
| Tool | Description | Permission |
|------|-------------|------------|
| `list_models` | List the authenticated user's effective models; filter with `kind=all`, `custom`, or `base` | Any |
| `get_model` | Get model configuration | Any |
| `create_model` | Create custom model | Admin |
| `update_model` | Update model settings and access grants while preserving the existing model form | Admin |
| `update_model_access` | Set grants for a custom, provider, or base model | Admin |
| `delete_model` | Delete a model | Admin |

`list_models` defaults to the authenticated user's `/api/models` response. Its
optional filters are `kind` (`all`, `custom`, or `base`), exact `provider` and
`connection_id`, plus substring `query`, `model_id`, `display_name`, and
`status` (`active` or `inactive`). Custom-vs-base classification uses the
Workspace model export, but never expands the user-scoped result set. The
separate admin-only `update_model_access` tool accepts Open WebUI's
`access_grants` for custom, provider, and base model IDs; `name` may be needed
when creating a provider/base access record. It is not available in the member
profile.

### Knowledge Base Management
| Tool | Description | Permission |
|------|-------------|------------|
| `list_knowledge_bases` | List knowledge bases | Any |
| `get_knowledge_base` | Get knowledge base details | Any |
| `create_knowledge_base` | Create knowledge base | Any |
| `update_knowledge_base` | Update knowledge base name/description | Owner |
| `update_knowledge_access` | Replace knowledge base access grants | Owner, write access, or admin |
| `delete_knowledge_base` | Delete knowledge base | Owner |

`update_knowledge_access` sends Open WebUI's exact access form,
`{"access_grants": [...]}`, to the dedicated knowledge access endpoint. Grant
entries use Open WebUI's native shape, such as
`{"principal_type": "group", "principal_id": "group-id", "permission": "read"}`. Open WebUI
applies the instance's sharing permissions and returns the updated knowledge
base, including its existing fields and files.

### File Management
| Tool | Description | Permission |
|------|-------------|------------|
| `upload_file` | Upload an absolute local file path, optionally linking it to a knowledge base | Any, subject to Open WebUI permissions |
| `list_files` | List uploaded files | Any |
| `search_files` | Search uploaded files by filename | Any |
| `get_file` | Get file metadata | Any |
| `get_file_content` | Get extracted file content | Any |
| `update_file_content` | Update extracted text content | File owner/write access |

### Chat Management
| Tool | Description | Permission |
|------|-------------|------------|
| `list_chats` | List user's chats | Own |
| `get_chat` | Get chat messages | Own |
| `delete_chat` | Delete a chat | Own |
| `delete_all_chats` | Delete all chats | Own |

### System
| Tool | Description | Permission |
|------|-------------|------------|
| `list_tools` | List available tools | Any |
| `list_functions` | List functions/filters | Any |
| `get_system_config` | Get full system config via Open WebUI's `/api/v1/configs/export` endpoint | Admin |

## Development

```bash
# Clone the repo
git clone https://github.com/troylar/open-webui-mcp-server.git
cd open-webui-mcp-server

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linter
ruff check src/
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or PR on GitHub.

## Related Projects

- [Open WebUI](https://github.com/open-webui/open-webui) - The web UI this server manages
- [FastMCP](https://github.com/jlowin/fastmcp) - The MCP framework used
- [MCPO](https://github.com/open-webui/mcpo) - MCP to OpenAPI proxy
