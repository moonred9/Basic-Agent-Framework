from __future__ import annotations

import heapq
import json
from pathlib import Path
from typing import Any, Protocol

from basic_agent_framework.tools.base import ToolResult, ToolSpec


class RetrieverBackend(Protocol):
    def retrieve(
        self,
        query: str,
        search_scope: list[str],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        ...


class ColPaliRetrieverBackend:
    def __init__(
        self,
        model_path: str = "model/colqwen2-v1.0-merged",
        device: str = "cuda:0",
        batch_size: int = 8,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def retrieve(
        self,
        query: str,
        search_scope: list[str],
        *,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query must be a non-empty string")
        if not search_scope:
            raise ValueError("search_scope must be a non-empty list of .pt paths")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        torch = self._import_torch()
        self._ensure_model_loaded()

        query_embeddings = self._embed_queries([query.strip()])
        corpus = self._load_scope_embeddings(search_scope, torch)
        scores = self._score(query_embeddings, corpus["embeddings"])
        score_list = scores[0] if scores else []

        top_indices = heapq.nlargest(
            min(top_k, len(score_list)),
            range(len(score_list)),
            key=lambda index: float(score_list[index]),
        )
        results: list[dict[str, Any]] = []
        for rank, page_index in enumerate(top_indices, start=1):
            ref = corpus["refs"][page_index]
            results.append(
                {
                    "rank": rank,
                    "score": float(score_list[page_index]),
                    "embedding_path": ref["embedding_path"],
                    "page_index": ref["page_index"],
                }
            )
        return results

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        torch = self._import_torch()
        try:
            from colpali_engine.models import ColQwen2, ColQwen2Processor
        except ImportError as exc:
            raise ImportError(
                "ColPaliRetrieverBackend requires the 'colpali_engine' package."
            ) from exc

        self._model = ColQwen2.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            local_files_only=True,
        ).eval()
        self._processor = ColQwen2Processor.from_pretrained(self.model_path)

    def _import_torch(self):
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "ColPaliRetrieverBackend requires the 'torch' package."
            ) from exc
        return torch

    def _embed_queries(self, queries: list[str]) -> list[Any]:
        embeddings: list[Any] = []
        assert self._model is not None
        assert self._processor is not None

        for start in range(0, len(queries), self.batch_size):
            batch_queries = queries[start : start + self.batch_size]
            inputs = self._processor.process_queries(batch_queries).to(self._model.device)
            with self._import_torch().no_grad():
                output = self._model(**inputs).cpu()
            embeddings.extend(output)
        return embeddings

    def _load_scope_embeddings(
        self,
        search_scope: list[str],
        torch_module: Any,
    ) -> dict[str, list[Any]]:
        merged_embeddings: list[Any] = []
        refs: list[dict[str, Any]] = []

        for embedding_path in search_scope:
            loaded = torch_module.load(embedding_path)
            page_embeddings = self._ensure_embedding_list(loaded)
            for page_index, embedding in enumerate(page_embeddings):
                merged_embeddings.append(embedding)
                refs.append(
                    {
                        "embedding_path": str(Path(embedding_path)),
                        "page_index": page_index,
                    }
                )

        if not merged_embeddings:
            raise ValueError("search_scope did not contain any page embeddings")
        return {"embeddings": merged_embeddings, "refs": refs}

    def _ensure_embedding_list(self, embeddings: Any) -> list[Any]:
        torch = self._import_torch()
        if isinstance(embeddings, list):
            return embeddings
        if isinstance(embeddings, torch.Tensor):
            if embeddings.ndim == 0:
                return [embeddings]
            return [embeddings[index] for index in range(embeddings.shape[0])]
        raise TypeError(f"Unsupported embedding container type: {type(embeddings)}")

    def _score(self, query_embeddings: list[Any], page_embeddings: list[Any]) -> list[list[float]]:
        assert self._processor is not None
        scores: list[list[float]] = []
        for query_start in range(0, len(query_embeddings), self.batch_size):
            batch_query = query_embeddings[query_start : query_start + self.batch_size]
            for page_start in range(0, len(page_embeddings), self.batch_size):
                batch_page = page_embeddings[page_start : page_start + self.batch_size]
                batch_scores = self._processor.score_multi_vector(batch_query, batch_page)
                for query_offset, row in enumerate(batch_scores):
                    target_index = query_start + query_offset
                    if target_index >= len(scores):
                        scores.append([])
                    scores[target_index].extend(float(value) for value in row.tolist())
        return scores


class RetrieveTool:
    name = "retrieve"
    description = "Retrieve relevant pages from embedding files using a ColPali query embedding."

    def __init__(
        self,
        backend: RetrieverBackend | None = None,
        default_top_k: int = 5,
    ) -> None:
        self.backend = backend or ColPaliRetrieverBackend()
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
                        "description": "Natural language retrieval query.",
                    },
                    "search_scope": {
                        "type": "array",
                        "description": "List of .pt embedding file paths to search over.",
                        "items": {"type": "string"},
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of retrieved pages.",
                        "default": self.default_top_k,
                    },
                },
                "required": ["query", "search_scope"],
            },
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        search_scope = arguments.get("search_scope")
        top_k = arguments.get("top_k", self.default_top_k)

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not isinstance(search_scope, list) or not search_scope:
            raise ValueError("search_scope must be a non-empty list of .pt paths")
        if not all(isinstance(path, str) and path.strip() for path in search_scope):
            raise ValueError("search_scope must contain only non-empty string paths")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        results = self.backend.retrieve(
            query.strip(),
            [path.strip() for path in search_scope],
            top_k=top_k,
        )
        content = "\n".join(
            [
                f"Query: {query.strip()}",
                f"Search scope size: {len(search_scope)}",
                "Results:",
                json.dumps(results, ensure_ascii=False, indent=2),
            ]
        )
        return ToolResult(
            name=self.name,
            content=content,
            metadata={
                "query": query.strip(),
                "search_scope": [path.strip() for path in search_scope],
                "result_count": len(results),
            },
        )
