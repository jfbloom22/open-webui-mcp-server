FROM ghcr.io/astral-sh/uv:0.10.7 AS uv

FROM python:3.12-slim

ARG SOURCE_SHA=unknown
ARG VERSION=production
ARG CREATED=unknown

WORKDIR /app

 # Install curl for the healthcheck and create a non-root runtime user.
 RUN apt-get update \
     && apt-get install -y --no-install-recommends curl \
     && rm -rf /var/lib/apt/lists/* \
     && useradd --uid 65532 --create-home --home-dir /home/openwebui-mcp --shell /usr/sbin/nologin openwebui-mcp

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

LABEL org.opencontainers.image.source="https://github.com/jfbloom22/open-webui-mcp-server" \
      org.opencontainers.image.revision="${SOURCE_SHA}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${CREATED}"

# Environment variables
ENV PYTHONUNBUFFERED=1
# Codex launches the local MCP server over stdio. Override this at runtime
# with MCP_TRANSPORT=http when an HTTP deployment is explicitly required.
ENV MCP_TRANSPORT=stdio
ENV MCP_HTTP_HOST=0.0.0.0
ENV MCP_HTTP_PORT=8000
ENV MCP_HTTP_PATH=/mcp

USER 65532:65532

# Expose MCP port
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=20s \
    CMD curl --silent --output /dev/null http://localhost:8000/mcp

CMD ["python", "-m", "src.openwebui_mcp.main"]
