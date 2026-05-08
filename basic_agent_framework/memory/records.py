from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


MemoryRecordType = Literal[
    "system",
    "user_input",
    "llm_output",
    "tool_call",
    "tool_result",
]


@dataclass(slots=True)
class MemoryRecord:
    type: MemoryRecordType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
