"""The :class:`Council` facade — the public entry point to the engine.

It owns the composed collaborators (event bus, moderator, scheduler, memory,
speech service) and exposes a tiny surface: feed it a user message, and it runs
the whole two-phase discussion, emitting events along the way. It deliberately
knows nothing about the CLI, voices or LEDs — those are subscribers.
"""

from __future__ import annotations

from app.agents.base import BaseAgent
from app.council.moderator import Moderator
from app.council.scheduler import Scheduler
from app.events.event_bus import EventBus
from app.events.event_types import ConversationEnded, UserMessage
from app.memory.history import History
from app.speech.service import SpeechService


class Council:
    """Coordinates a group of agents into a live, event-driven discussion."""

    def __init__(
        self,
        *,
        bus: EventBus,
        agents: list[BaseAgent],
        moderator: Moderator,
        scheduler: Scheduler,
        history: History,
        speech_service: SpeechService,
        default_conversation_id: str = "main",
    ) -> None:
        self._bus = bus
        self._agents = agents
        self._moderator = moderator
        self._scheduler = scheduler
        self._history = history
        self._speech = speech_service
        self._default_conversation_id = default_conversation_id
        # Materialize the default conversation eagerly so it always exists.
        self._history.get_or_create(default_conversation_id)

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def members(self) -> list[BaseAgent]:
        return list(self._agents)

    async def ask(self, text: str, *, conversation_id: str | None = None) -> None:
        """Submit a user message and run the discussion it triggers."""
        conversation = self._history.get_or_create(conversation_id or self._default_conversation_id)
        conversation.add_user(text)
        await self._bus.publish(UserMessage(conversation_id=conversation.id, text=text))
        await self._scheduler.run_turn(conversation)

    async def shutdown(self, *, conversation_id: str | None = None, reason: str = "session ended") -> None:
        """Announce the conversation end and release speech resources."""
        cid = conversation_id or self._default_conversation_id
        await self._bus.publish(ConversationEnded(conversation_id=cid, reason=reason))
        await self._speech.aclose()
