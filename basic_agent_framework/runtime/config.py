from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AgentConfig:
    max_steps: int = 10
    require_tool_call: bool = True
    stop_tool_name: str = "answer"
