"""The speech backend abstraction.

A :class:`SpeechBackend` turns a :class:`~app.events.event_types.SpeakEvent`
into a *stream* of :class:`SpeechChunk` objects. Streaming (rather than a single
blocking ``say(text)``) is the key future-proofing decision:

* Today the terminal backend yields one chunk per word with a synthesized
  amplitude.
* Tomorrow the ElevenLabs backend yields chunks as audio arrives, carrying real
  amplitude envelopes and raw PCM bytes.

The :class:`~app.speech.service.SpeechService` consumes this stream and
translates chunks into ``SpeechProgress`` events — which is precisely what the
Raspberry Pi LED lamp will subscribe to. No backend ever knows LEDs exist.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from pydantic import BaseModel, ConfigDict

from app.events.event_types import SpeakEvent


class SpeechChunk(BaseModel):
    """One increment of rendered speech."""

    model_config = ConfigDict(frozen=True)

    # Normalized loudness in ``0.0..1.0`` — drives LED brightness later.
    amplitude: float = 0.0
    # The slice of text this chunk covers (word/sentence), if any.
    text_segment: str | None = None
    # Raw audio bytes for this chunk (populated by real TTS backends only).
    audio: bytes | None = None


class SpeechBackend(ABC):
    """Renders spoken lines. Implementations must be safe to call repeatedly."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs (e.g. ``"terminal"``)."""

    @abstractmethod
    def stream(self, event: SpeakEvent) -> AsyncIterator[SpeechChunk]:
        """Yield :class:`SpeechChunk` objects for ``event``.

        Implemented as an ``async def`` generator. Must not raise for empty
        text; yield nothing instead.
        """

    async def aclose(self) -> None:
        """Release any resources (sockets, audio devices). Default: no-op."""
        return None
