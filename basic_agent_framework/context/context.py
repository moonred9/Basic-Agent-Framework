from __future__ import annotations

from typing import TYPE_CHECKING

from basic_agent_framework.context.adapters import ContextAdapter, TaggedContextAdapter
from basic_agent_framework.context.message import Message
from basic_agent_framework.memory.memory import Memory
from basic_agent_framework.memory.records import MemoryRecord
from basic_agent_framework.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from basic_agent_framework.agent.state import AgentState


DEFAULT_SYSTEM_PROMPT = (
    "You are an agent that can iteratively search for evidence and answer the user. "
    "Do not answer directly outside a tool call."
)


class Context:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        adapter: ContextAdapter | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        memory_limit: int | None = None,
    ) -> None:
        self.adapter = adapter or TaggedContextAdapter()
        self.system_prompt = system_prompt
        self.tool_registry = tool_registry
        self.memory_limit = memory_limit

    def build_messages(self, memory: Memory, state: AgentState) -> list[Message]:
        tool_instructions = self.adapter.render_tool_instructions(
            self.tool_registry.list_specs()
        )
        system_content = "\n\n".join(
            [
                self.system_prompt,
                tool_instructions,
                f"Current step: {state.current_step + 1}/{state.max_steps}.",
            ]
        )
        messages = [self.adapter.render_system_message(system_content)]

        for record in memory.get_context_records(limit=self.memory_limit):
            messages.extend(self._render_record(record))

        return messages

    def _render_record(self, record: MemoryRecord) -> list[Message]:
        if record.type == "system":
            return [self.adapter.render_system_message(record.content)]
        if record.type == "user_input":
            return [self.adapter.render_user_message(record.content)]
        if record.type == "llm_output":
            return [self.adapter.render_assistant_message(record.content)]
        if record.type == "tool_result":
            tool_name = str(record.metadata.get("tool_name", "tool"))
            return [self.adapter.render_tool_response(tool_name, record.content)]
        if record.type == "tool_call":
            return []
        return []
