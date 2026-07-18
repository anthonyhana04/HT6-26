"""The base :class:`Event` type all events inherit from."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Event(BaseModel):
    """Common envelope for everything that flows across the event bus.

    Events are immutable value objects. Concrete events add their own payload
    fields. The class name doubles as the topic used by :class:`~app.events.
    event_bus.EventBus` for subscription, so subclasses need no registration.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def name(self) -> str:
        """Human/topic-friendly name of the event (its class name)."""
        return type(self).__name__
