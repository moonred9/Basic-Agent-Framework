from __future__ import annotations

from uuid import uuid4

from basic_agent_framework.agent.state import AgentResult, AgentState
from basic_agent_framework.context.context import Context
from basic_agent_framework.llm.client import LLMClient
from basic_agent_framework.llm.output_parser import OutputParser
from basic_agent_framework.memory.memory import Memory
from basic_agent_framework.runtime.config import AgentConfig
from basic_agent_framework.runtime.errors import MaxStepsExceededError
from basic_agent_framework.tools.registry import ToolRegistry


class Agent:
    def __init__(
        self,
        llm_client: LLMClient,
        context: Context,
        memory: Memory,
        tool_registry: ToolRegistry,
        config: AgentConfig | None = None,
        output_parser: OutputParser | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.context = context
        self.memory = memory
        self.tool_registry = tool_registry
        self.config = config or AgentConfig()
        self.output_parser = output_parser or OutputParser()

    def run(self, user_input: str) -> AgentResult:
        state = AgentState(
            run_id=uuid4().hex,
            user_input=user_input,
            max_steps=self.config.max_steps,
        )
        self.memory.add_user_input(user_input)

        while state.current_step < state.max_steps:
            messages = self.context.build_messages(self.memory, state)
            response = self.llm_client.generate(messages)
            state.last_llm_output = response.content
            self.memory.add_llm_output(response.content)

            parsed = self.output_parser.parse(response)
            tool_call = parsed.tool_call
            if tool_call is None:
                raise MaxStepsExceededError("LLM response did not contain a tool call.")

            state.last_tool_call = tool_call
            self.memory.add_tool_call(tool_call)
            tool_result = self.tool_registry.run(tool_call)
            self.memory.add_tool_result(tool_result)
            state.current_step += 1

            if tool_call.name == self.config.stop_tool_name:
                state.finished = True
                state.answer = tool_result.content
                return AgentResult(
                    answer=tool_result.content,
                    steps=state.current_step,
                    finished=True,
                    run_id=state.run_id,
                    metadata={"last_tool": tool_call.name},
                )

        raise MaxStepsExceededError(
            f"Agent reached max_steps={state.max_steps} without calling "
            f"{self.config.stop_tool_name}."
        )
