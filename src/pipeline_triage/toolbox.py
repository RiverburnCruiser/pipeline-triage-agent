"""The agent's view of the MCP server: tool definitions in, tool results out.

The agent never imports the store or the runbook index. It only sees what the
MCP server exposes, so the same agent works against any server with these tools.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from mcp import Client, StdioServerParameters
from mcp.types import TextContent


@dataclass(frozen=True)
class ToolResult:
    text: str
    """What the model sees: the structured result as JSON, or the error message."""
    data: dict[str, Any] | None
    """The structured result, for code that needs fields rather than text."""
    is_error: bool


class McpToolbox:
    def __init__(self, client: Client) -> None:
        self._client = client
        self._definitions: list[dict[str, Any]] = []

    async def load(self) -> McpToolbox:
        tools = (await self._client.list_tools()).tools
        self._definitions = [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in tools
        ]
        return self

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """Tool definitions in the shape the Claude Messages API expects."""
        return self._definitions

    @property
    def names(self) -> set[str]:
        return {d["name"] for d in self._definitions}

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = await self._client.call_tool(name, arguments)
        data = result.structured_content
        if data is not None and not result.is_error:
            text = json.dumps(data, separators=(",", ":"))
        else:
            text = "\n".join(c.text for c in result.content if isinstance(c, TextContent))
        return ToolResult(text=text, data=data, is_error=bool(result.is_error))

    async def failed_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Failed runs, newest first, as get_failed_runs returns them."""
        result = await self.call("get_failed_runs", {"limit": limit})
        if result.is_error:
            raise RuntimeError(result.text)
        return list((result.data or {}).get("result", []))


def stdio_server() -> StdioServerParameters:
    """Run this package's MCP server as a subprocess, the way Claude Code does."""
    return StdioServerParameters(
        command=sys.executable, args=["-m", "pipeline_triage.mcp_server"], env=dict(os.environ)
    )
