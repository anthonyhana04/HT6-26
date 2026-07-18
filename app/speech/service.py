"""The speech layer's bus adapter.

The :class:`SpeechService` is the *only* thing that knows both about the event
bus and a concrete :class:`SpeechBackend`. It subscribes to ``SpeakEvent`` and
translates a backend's chunk stream into the lifecycle events the rest of the
system (and future hardware) reacts to:

    SpeakEvent  →  SpeechStarted  →  SpeechProgress*  →  SpeechFinished

The Raspberry Pi LED lamp will later subscribe to exactly these three events.
Because the council awaits ``SpeechFinished`` (via ``EventBus.wait_for``), this
adapter also guarantees the "no overlapping speakers" rule.
"""

from __future__ import annotations

import logging

from app.events.event_bus import EventBus
from app.events.event_types import (
    SpeakEvent,
    SpeechFinished,
    SpeechProgress,
    SpeechStarted,
)
from app.speech.base import SpeechBackend

logger = logging.getLogger("ai_council.speech")


class SpeechService:
    """Bridges ``SpeakEvent`` to a :class:`SpeechBackend` and back to events."""

    def __init__(self, bus: EventBus, backend: SpeechBackend) -> None:
        self._bus = bus
        self._backend = backend
        self._subscription = bus.subscribe(SpeakEvent, self._on_speak)

    @property
    def backend(self) -> SpeechBackend:
        return self._backend

    async def _on_speak(self, event: SpeakEvent) -> None:
        await self._bus.publish(
            SpeechStarted(
                conversation_id=event.conversation_id,
                speaker=event.speaker,
                correlation_id=event.correlation_id,
            )
        )
        try:
            async for chunk in self._backend.stream(event):
                await self._bus.publish(
                    SpeechProgress(
                        conversation_id=event.conversation_id,
                        speaker=event.speaker,
                        correlation_id=event.correlation_id,
                        amplitude=chunk.amplitude,
                        text_segment=chunk.text_segment,
                    )
                )
        except Exception:  # noqa: BLE001 — never let a backend fault strand a turn
            logger.exception("Speech backend %r failed for %s", self._backend.name, event.speaker)
        finally:
            await self._bus.publish(
                SpeechFinished(
                    conversation_id=event.conversation_id,
                    speaker=event.speaker,
                    correlation_id=event.correlation_id,
                )
            )

    async def aclose(self) -> None:
        self._subscription.cancel()
        await self._backend.aclose()
