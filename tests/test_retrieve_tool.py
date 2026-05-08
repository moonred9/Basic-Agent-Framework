from __future__ import annotations

import unittest

from basic_agent_framework.tools.retrieve import RetrieveTool


class FakeRetrieverBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def retrieve(
        self,
        query: str,
        search_scope: list[str],
        *,
        top_k: int = 5,
    ) -> list[dict]:
        self.calls.append((query, search_scope, top_k))
        return [
            {
                "rank": 1,
                "score": 0.99,
                "embedding_path": search_scope[0],
                "page_index": 3,
            }
        ]


class RetrieveToolTest(unittest.TestCase):
    def test_retrieve_tool_returns_backend_results(self) -> None:
        backend = FakeRetrieverBackend()
        tool = RetrieveTool(backend=backend, default_top_k=2)

        result = tool.run(
            {
                "query": "find the relevant page",
                "search_scope": ["/tmp/doc_a.pt", "/tmp/doc_b.pt"],
            }
        )

        self.assertEqual(
            backend.calls,
            [("find the relevant page", ["/tmp/doc_a.pt", "/tmp/doc_b.pt"], 2)],
        )
        self.assertEqual(result.name, "retrieve")
        self.assertIn('"embedding_path": "/tmp/doc_a.pt"', result.content)
        self.assertEqual(result.metadata["result_count"], 1)

    def test_retrieve_tool_requires_search_scope(self) -> None:
        tool = RetrieveTool(backend=FakeRetrieverBackend())

        with self.assertRaises(ValueError):
            tool.run({"query": "x", "search_scope": []})


if __name__ == "__main__":
    unittest.main()
