"""Phase 1 output: a lightweight :class:`Proposal` to speak (or stay silent)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Intent(str, Enum):
    """The kind of contribution an agent intends to make.

    Intents drive moderator scheduling (e.g. a ``CORRECTION`` is ordered after
    its target, a ``SUMMARY`` tends to close a discussion).
    """

    ANSWER = "ANSWER"
    QUESTION = "QUESTION"
    CORRECTION = "CORRECTION"
    DISAGREEMENT = "DISAGREEMENT"
    AGREEMENT = "AGREEMENT"
    FOLLOW_UP = "FOLLOW_UP"
    SUMMARY = "SUMMARY"
    OBSERVATION = "OBSERVATION"


class Proposal(BaseModel):
    """An agent's bid to participate, produced cheaply in the proposal phase.

    Crucially this contains *no* generated prose — only the metadata the
    moderator needs to decide who speaks. This is what keeps the proposal
    phase token-light.
    """

    model_config = ConfigDict(frozen=True)

    agent: str
    should_speak: bool
    confidence: int = Field(ge=0, le=100)
    intent: Intent
    reason: str
    target: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: object) -> int:
        """Be forgiving with model output: coerce and clamp to ``0..100``."""
        try:
            number = int(round(float(value)))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0
        return max(0, min(100, number))

    @property
    def is_bid(self) -> bool:
        """True when the agent actually wants the floor."""
        return self.should_speak
