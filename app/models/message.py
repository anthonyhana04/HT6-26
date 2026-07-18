"""The atomic unit of conversation memory: a :class:`Message`."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, Enum):
    """Who authored a message."""

    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class Message(BaseModel):
    """A single utterance in a conversation.

    Messages are immutable once created. ``speaker`` is a human-readable name
    ("User", "DeepSeek", "Anthropic", ...) so transcripts and future TTS routing
    do not need to reach back into agent objects.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: uuid4().hex)
    role: Role
    speaker: str
    content: str
    created_at: datetime = Field(default_factory=_utcnow)

    # Optional discussion metadata (present on agent messages).
    intent: str | None = None
    target: str | None = None

    def as_transcript_line(self) -> str:
        """Render a compact ``Speaker: content`` line for prompt context."""
        prefix = self.speaker
        if self.target:
            prefix = f"{self.speaker} → {self.target}"
        return f"{prefix}: {self.content}"
