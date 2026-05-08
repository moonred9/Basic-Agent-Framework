from __future__ import annotations

from typing import Any, Protocol

from basic_agent_framework.tools.base import ToolResult, ToolSpec


class SearchBackend(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> list[str]:
        ...


class StaticSearchBackend:
    def __init__(self, documents: list[str] | None = None) -> None:
        self.documents = documents or []

    def search(self, query: str, *, top_k: int = 5) -> list[str]:
        query_terms = {term.lower() for term in query.split() if term.strip()}
        if not query_terms:
            return self.documents[:top_k]

        scored: list[tuple[int, str]] = []
        for document in self.documents:
            lower_document = document.lower()
            score = sum(1 for term in query_terms if term in lower_document)
            if score:
                scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[:top_k]]


class SearchTool:
    name = "search"
    description = "Search external knowledge and return relevant evidence."

    def __init__(self, backend: SearchBackend | None = None, default_top_k: int = 5) -> None:
        self.backend = backend or StaticSearchBackend()
        self.default_top_k = default_top_k

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of search results.",
                        "default": self.default_top_k,
                    },
                },
                "required": ["query"],
            },
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        top_k = arguments.get("top_k", self.default_top_k)
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        results = self.backend.search(query.strip(), top_k=top_k)
        if not results:
            content = f"Query: {query.strip()}\nNo results found."
        else:
            lines = [f"Query: {query.strip()}", "Results:"]
            lines.extend(f"{index}. {result}" for index, result in enumerate(results, start=1))
            content = "\n".join(lines)

        return ToolResult(
            name=self.name,
            content=content,
            metadata={"query": query.strip(), "result_count": len(results)},
        )
