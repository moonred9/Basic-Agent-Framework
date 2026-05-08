from __future__ import annotations

from typing import Protocol

from basic_agent_framework.context.message import Message
from basic_agent_framework.llm.schemas import LLMResponse


class LLMClient(Protocol):
    def generate(self, messages: list[Message]) -> LLMResponse:
        ...


class OpenAILLMClient:
    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAILLMClient requires the 'openai' package to be installed."
            ) from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, messages: list[Message]) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "messages": [message.to_openai_dict() for message in messages],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        return LLMResponse(content=content, raw=response)
