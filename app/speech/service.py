"""The speech layer's bus adapter.

The :class:`SpeechService` is the *only* thing that knows both about the event
bus and a concrete :class:`SpeechBackend`. It subscribes to ``SpeakEvent`` and
translates a backend's chunk stream into the lifecycle events the rest of the
system (and future hardware) reacts to:

    SpeakEvent  →  SpeechStarted  →  SpeechProgress*  →  SpeechFinished

``UserBargeIn`` aborts the current stream early so the user can cut the council
off mid-sentence. ``SpeechFinished`` still fires so the scheduler unblocks.
"""

from __future__ import annotations

import asyncio
import logging

from app.events.event_bus import EventBus
from app.events.event_types import (
    SpeakEvent,
    SpeechFinished,
    SpeechProgress,
    SpeechStarted,
    UserBargeIn,
)
from app.speech.base import SpeechBackend

logger = logging.getLogger("ai_council.speech")


class SpeechService:
    """Bridges ``SpeakEvent`` to a :class:`SpeechBackend` and back to events."""

    def __init__(self, bus: EventBus, backend: SpeechBackend) -> None:
        self._bus = bus
        self._backend = backend
        self._cancel = asyncio.Event()
        self._subscription = bus.subscribe(SpeakEvent, self._on_speak)
        self._barge_sub = bus.subscribe(UserBargeIn, self._on_barge_in)

    @property
    def backend(self) -> SpeechBackend:
        return self._backend

    def request_stop(self) -> None:
        """Abort the in-flight speak stream (if any)."""
        self._cancel.set()
        abort = getattr(self._backend, "abort", None)
        if abort is not None:
            try:
                abort()
            except Exception:  # noqa: BLE001
                pass

    def _on_barge_in(self, event: UserBargeIn) -> None:
        logger.info("Barge-in: %s", event.reason)
        self.request_stop()

    async def _on_speak(self, event: SpeakEvent) -> None:
        self._cancel.clear()
        await self._bus.publish(
            SpeechStarted(
                conversation_id=event.conversation_id,
                speaker=event.speaker,
                correlation_id=event.correlation_id,
            )
        )
        try:
            async for chunk in self._backend.stream(event):
                if self._cancel.is_set():
                    logger.info("Speech cut short for %s (barge-in)", event.speaker)
                    break
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
        self._cancel.set()
        self._subscription.cancel()
        self._barge_sub.cancel()
        await self._backend.aclose()
