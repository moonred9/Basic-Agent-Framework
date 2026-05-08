from __future__ import annotations

import unittest

from basic_agent_framework.llm.output_parser import OutputParser
from basic_agent_framework.llm.schemas import LLMResponse
from basic_agent_framework.runtime.errors import ParseError


class OutputParserTest(unittest.TestCase):
    def test_parse_tagged_tool_call(self) -> None:
        parsed = OutputParser().parse(
            LLMResponse(
                content=(
                    "<think>Need search.</think>\n"
                    '<action>{"name": "search", "arguments": {"query": "x"}}</action>'
                )
            )
        )

        self.assertEqual(parsed.reasoning, "Need search.")
        self.assertIsNotNone(parsed.tool_call)
        self.assertEqual(parsed.tool_call.name, "search")
        self.assertEqual(parsed.tool_call.arguments, {"query": "x"})

    def test_missing_tool_call_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError):
            OutputParser().parse(LLMResponse(content="<think>No action.</think>"))


if __name__ == "__main__":
    unittest.main()
