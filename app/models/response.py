"""Phase 2 output: the full :class:`Response` an accepted speaker generates."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

from app.models.proposal import Intent


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Response(BaseModel):
    """The actual spoken contribution of a council member.

    A response is generated only *after* the moderator accepts the agent's
    proposal, so exactly one round-trip of heavy generation happens per
    scheduled speaker.
    """

    model_config = ConfigDict(frozen=True)

    agent: str
    text: str
    intent: Intent
    target: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
