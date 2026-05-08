from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from basic_agent_framework.tools.base import ToolCall


@dataclass(slots=True)
class LLMResponse:
    content: str
    raw: Any | None = None


@dataclass(slots=True)
class ParsedOutput:
    reasoning: str | None
    tool_call: ToolCall | None
    raw_text: str
