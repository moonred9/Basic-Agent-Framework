from __future__ import annotations

import ast
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
        action_tag: str = "action",
    ) -> None:
        self.reasoning_tag = reasoning_tag
        self.action_tag = action_tag

    def parse(self, response: LLMResponse) -> ParsedOutput:
        text = response.content
        reasoning = self._extract_tag(text, self.reasoning_tag)
        action_text = self._extract_tag(text, self.action_tag)
        if action_text is None:
            raise ParseError(f"Missing <{self.action_tag}> action.")

        payload = self._parse_json(action_text)
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if not isinstance(name, str) or not name.strip():
            raise ParseError("Action must include a non-empty string name.")
        if not isinstance(arguments, dict):
            raise ParseError("Action arguments must be an object.")

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
            try:
                payload = ast.literal_eval(content)
            except (SyntaxError, ValueError) as fallback_exc:
                raise ParseError(f"Invalid action JSON: {exc}") from fallback_exc
        if not isinstance(payload, dict):
            raise ParseError("Action JSON must be an object.")
        return payload
