from __future__ import annotations

from typing import Any

from basic_agent_framework.tools.base import ToolResult, ToolSpec


class AnswerTool:
    name = "answer"
    description = "Return the final answer and stop the agent loop."

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to return to the user.",
                    }
                },
                "required": ["answer"],
            },
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        answer = arguments.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be a non-empty string")
        return ToolResult(name=self.name, content=answer.strip())
