from __future__ import annotations

import os

from basic_agent_framework import (
    Agent,
    AgentConfig,
    AnswerTool,
    Context,
    InMemoryMemory,
    OpenAILLMClient,
    SearchTool,
    StaticSearchBackend,
    ToolRegistry,
)


def main() -> None:
    registry = ToolRegistry(
        [
            SearchTool(
                StaticSearchBackend(
                    [
                        "The framework supports search and answer tools.",
                        "Calling the answer tool stops the agent loop.",
                    ]
                )
            ),
            AnswerTool(),
        ]
    )
    agent = Agent(
        llm_client=OpenAILLMClient(
            model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=0,
        ),
        context=Context(tool_registry=registry),
        memory=InMemoryMemory(),
        tool_registry=registry,
        config=AgentConfig(max_steps=10),
    )

    result = agent.run("这个框架现在有哪些工具？")
    print(result.answer)


if __name__ == "__main__":
    main()
