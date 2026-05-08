from __future__ import annotations

import unittest

from basic_agent_framework import (
    Agent,
    AgentConfig,
    AnswerTool,
    Context,
    InMemoryMemory,
    SearchTool,
    StaticSearchBackend,
    ToolRegistry,
)
from basic_agent_framework.context.message import Message
from basic_agent_framework.llm.schemas import LLMResponse


class FakeLLMClient:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[list[Message]] = []

    def generate(self, messages: list[Message]) -> LLMResponse:
        self.calls.append(messages)
        if not self.outputs:
            raise AssertionError("FakeLLMClient has no remaining outputs")
        return LLMResponse(content=self.outputs.pop(0))


def build_agent(llm_client: FakeLLMClient, memory: InMemoryMemory | None = None) -> Agent:
    registry = ToolRegistry(
        [
            AnswerTool(),
            SearchTool(
                StaticSearchBackend(
                    [
                        "Python is a programming language.",
                        "OpenAI provides APIs for language models.",
                    ]
                )
            ),
        ]
    )
    return Agent(
        llm_client=llm_client,
        context=Context(tool_registry=registry),
        memory=memory or InMemoryMemory(),
        tool_registry=registry,
        config=AgentConfig(max_steps=3),
    )


class AgentLoopTest(unittest.TestCase):
    def test_answer_tool_stops_loop(self) -> None:
        llm = FakeLLMClient(
            [
                '<think>Enough info.</think>\n'
                '<action>{"name": "answer", "arguments": {"answer": "final"}}</action>'
            ]
        )
        agent = build_agent(llm)

        result = agent.run("question")

        self.assertTrue(result.finished)
        self.assertEqual(result.answer, "final")
        self.assertEqual(result.steps, 1)
        self.assertEqual(len(llm.calls), 1)

    def test_search_result_is_added_to_next_context(self) -> None:
        memory = InMemoryMemory()
        llm = FakeLLMClient(
            [
                '<think>Need evidence.</think>\n'
                '<action>{"name": "search", "arguments": {"query": "OpenAI APIs"}}</action>',
                '<think>Now answer.</think>\n'
                '<action>{"name": "answer", "arguments": {"answer": "OpenAI provides APIs."}}</action>',
            ]
        )
        agent = build_agent(llm, memory)

        result = agent.run("What does OpenAI provide?")

        self.assertEqual(result.answer, "OpenAI provides APIs.")
        self.assertEqual(result.steps, 2)
        self.assertEqual(len(llm.calls), 2)
        second_context = "\n".join(message.content for message in llm.calls[1])
        self.assertIn("OpenAI provides APIs for language models.", second_context)


if __name__ == "__main__":
    unittest.main()
