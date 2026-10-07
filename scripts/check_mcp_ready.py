#!/usr/bin/env python3
"""Verify an authenticated MCP connection using the FastMCP client SDK."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from fastmcp import Client


async def check_server(url: str, token: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not return the expected tool"

    while time.monotonic() < deadline:
        try:
            async with Client(url, auth=token, timeout=5, init_timeout=5) as client:
                tools = await client.list_tools()
                if any(tool.name == "list_models" for tool in tools):
                    return
                last_error = "list_models is missing from the MCP tool catalog"
        except Exception as error:
            last_error = type(error).__name__

        await asyncio.sleep(0.5)

    raise RuntimeError(f"MCP readiness check did not succeed: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Streamable HTTP MCP endpoint")
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()
    token = os.environ.get("MCP_HTTP_TOKEN", "")
    if not token:
        print("MCP_HTTP_TOKEN is required", file=sys.stderr)
        return 2

    try:
        asyncio.run(check_server(args.url, token, args.timeout))
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
