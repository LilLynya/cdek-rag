from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="Сообщение пользователя.")
    session_id: Optional[str] = Field(
        default=None,
        description="Идентификатор диалога. Если не передан — будет сгенерирован.",
    )


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    needs_clarification: bool = False
    sources: list[str] = Field(default_factory=list)
