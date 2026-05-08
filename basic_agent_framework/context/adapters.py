from __future__ import annotations

import json
import textwrap
from typing import Protocol

from basic_agent_framework.context.message import Message
from basic_agent_framework.tools.base import ToolSpec


class ContextAdapter(Protocol):
    def render_system_message(self, content: str) -> Message:
        ...

    def render_user_message(self, content: str) -> Message:
        ...

    def render_assistant_message(self, content: str) -> Message:
        ...

    def render_tool_response(self, tool_name: str, content: str) -> Message:
        ...

    def render_tool_instructions(self, tools: list[ToolSpec]) -> str:
        ...


class TaggedContextAdapter:
    reasoning_tag = "think"
    action_tag = "action"
    tool_response_tag = "tool_response"

    def render_system_message(self, content: str) -> Message:
        return Message(role="system", content=content)

    def render_user_message(self, content: str) -> Message:
        return Message(role="user", content=content)

    def render_assistant_message(self, content: str) -> Message:
        return Message(role="assistant", content=content)

    def render_tool_response(self, tool_name: str, content: str) -> Message:
        wrapped = (
            f"<{self.tool_response_tag} name=\"{tool_name}\">\n"
            f"{content}\n"
            f"</{self.tool_response_tag}>"
        )
        return Message(role="user", content=wrapped, name=tool_name)

    def render_tool_instructions(self, tools: list[ToolSpec]) -> str:
        specs = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools
        ]
        examples = textwrap.dedent(
            """
            Example 1:
            <think>Need evidence from the search tool.</think>
            <action>{"name": "search", "arguments": {"query": "OpenAI APIs"}}</action>

            Example 2:
            <think>I have enough evidence to answer.</think>
            <action>{"name": "answer", "arguments": {"answer": "OpenAI provides APIs."}}</action>
            """
        ).strip()
        return (
            "You must respond with exactly one tool call per turn.\n"
            "Do not write Markdown fences. Do not write any text outside the allowed tags.\n"
            f"First write optional reasoning inside <{self.reasoning_tag}>...</{self.reasoning_tag}>.\n"
            f"Then write a JSON action inside <{self.action_tag}>...</{self.action_tag}>.\n"
            "The action JSON schema is: "
            '{"name": "<tool name>", "arguments": {...}}.\n'
            "Use the answer tool only when you are ready to return the final answer. "
            "Calling answer stops the agent loop.\n"
            "Copy the response format from these examples exactly.\n"
            f"{examples}\n"
            "Available tools:\n"
            f"{json.dumps(specs, ensure_ascii=False, indent=2)}"
        )
