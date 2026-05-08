from __future__ import annotations

import unittest

from basic_agent_framework.tools.answer import AnswerTool
from basic_agent_framework.tools.search import SearchTool, StaticSearchBackend


class ToolTest(unittest.TestCase):
    def test_answer_tool_returns_final_answer(self) -> None:
        result = AnswerTool().run({"answer": " done "})

        self.assertEqual(result.name, "answer")
        self.assertEqual(result.content, "done")

    def test_static_search_backend_returns_matching_documents(self) -> None:
        tool = SearchTool(StaticSearchBackend(["alpha beta", "gamma delta"]))

        result = tool.run({"query": "beta"})

        self.assertIn("alpha beta", result.content)
        self.assertEqual(result.metadata["result_count"], 1)


if __name__ == "__main__":
    unittest.main()
