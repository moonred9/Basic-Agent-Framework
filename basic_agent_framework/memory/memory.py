from __future__ import annotations

import json
from typing import Protocol

from basic_agent_framework.memory.records import MemoryRecord
from basic_agent_framework.tools.base import ToolCall, ToolResult


class Memory(Protocol):
    def add_system(self, content: str) -> None:
        ...

    def add_user_input(self, content: str) -> None:
        ...

    def add_llm_output(self, content: str) -> None:
        ...

    def add_tool_call(self, tool_call: ToolCall) -> None:
        ...

    def add_tool_result(self, result: ToolResult) -> None:
        ...

    def get_context_records(self, limit: int | None = None) -> list[MemoryRecord]:
        ...

    def clear(self) -> None:
        ...


class InMemoryMemory:
    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []

    def add_system(self, content: str) -> None:
        self.records.append(MemoryRecord(type="system", content=content))

    def add_user_input(self, content: str) -> None:
        self.records.append(MemoryRecord(type="user_input", content=content))

    def add_llm_output(self, content: str) -> None:
        self.records.append(MemoryRecord(type="llm_output", content=content))

    def add_tool_call(self, tool_call: ToolCall) -> None:
        self.records.append(
            MemoryRecord(
                type="tool_call",
                content=json.dumps(
                    {"name": tool_call.name, "arguments": tool_call.arguments},
                    ensure_ascii=False,
                ),
                metadata={"tool_name": tool_call.name},
            )
        )

    def add_tool_result(self, result: ToolResult) -> None:
        self.records.append(
            MemoryRecord(
                type="tool_result",
                content=result.content,
                metadata={"tool_name": result.name, **result.metadata},
            )
        )

    def get_context_records(self, limit: int | None = None) -> list[MemoryRecord]:
        if limit is None:
            return list(self.records)
        return list(self.records[-limit:])

    def clear(self) -> None:
        self.records.clear()
