from __future__ import annotations

import json
import re
from typing import Any

from basic_agent_framework.llm.schemas import LLMResponse, ParsedOutput
from basic_agent_framework.runtime.errors import ParseError
from basic_agent_framework.tools.base import ToolCall


class OutputParser:
    def __init__(
        self,
        *,
        reasoning_tag: str = "think",
        tool_call_tag: str = "tool_call",
    ) -> None:
        self.reasoning_tag = reasoning_tag
        self.tool_call_tag = tool_call_tag

    def parse(self, response: LLMResponse) -> ParsedOutput:
        text = response.content
        reasoning = self._extract_tag(text, self.reasoning_tag)
        tool_call_text = self._extract_tag(text, self.tool_call_tag)
        if tool_call_text is None:
            raise ParseError(f"Missing <{self.tool_call_tag}> tool call.")

        payload = self._parse_json(tool_call_text)
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise ParseError("Tool call must include a non-empty string name.")
        if not isinstance(arguments, dict):
            raise ParseError("Tool call arguments must be an object.")

        return ParsedOutput(
            reasoning=reasoning,
            tool_call=ToolCall(name=name.strip(), arguments=arguments),
            raw_text=text,
        )

    def _extract_tag(self, text: str, tag: str) -> str | None:
        pattern = rf"<{re.escape(tag)}(?:\s[^>]*)?>(.*?)</{re.escape(tag)}>"
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            return None
        return match.group(1).strip()

    def _parse_json(self, content: str) -> dict[str, Any]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Invalid tool call JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ParseError("Tool call JSON must be an object.")
        return payload
