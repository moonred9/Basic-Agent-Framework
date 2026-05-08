from __future__ import annotations

from basic_agent_framework.runtime.errors import ToolExecutionError, ToolNotFoundError
from basic_agent_framework.tools.base import Tool, ToolCall, ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(f"Tool is not registered: {name}") from exc

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools.values()]

    def run(self, tool_call: ToolCall) -> ToolResult:
        tool = self.get(tool_call.name)
        try:
            return tool.run(tool_call.arguments)
        except Exception as exc:
            raise ToolExecutionError(
                f"Tool failed: {tool_call.name}: {exc}"
            ) from exc
