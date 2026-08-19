FROM ghcr.io/astral-sh/uv:0.10.7 AS uv

FROM python:3.12-slim

WORKDIR /app

COPY --from=uv /uv /uvx /bin/

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install locked runtime dependencies before application code for efficient rebuilds.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

# Copy application code
COPY src/ ./src/
RUN uv sync --locked --no-dev

# Local MCP clients communicate over standard input/output by default.
CMD ["openwebui-mcp"]
