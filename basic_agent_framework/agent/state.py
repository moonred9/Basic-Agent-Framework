from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from basic_agent_framework.tools.base import ToolCall


@dataclass(slots=True)
class AgentState:
    run_id: str
    user_input: str
    current_step: int = 0
    max_steps: int = 10
    finished: bool = False
    last_llm_output: str | None = None
    last_tool_call: ToolCall | None = None
    answer: str | None = None


@dataclass(slots=True)
class AgentResult:
    answer: str
    steps: int
    finished: bool
    run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
