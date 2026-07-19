"""Speech-to-text: the input side of the speech layer.

Where a :class:`~app.speech.base.SpeechBackend` turns text into sound, a
:class:`Transcriber` turns sound into text. Keeping it behind an ABC means the
microphone front-end (a phone browser, a USB mic, anything) only ever produces
audio bytes and receives a string — it never knows which STT engine ran.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("ai_council.speech.stt")


def _content_type_for(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".wav"):
        return "audio/wav"
    if name.endswith(".webm"):
        return "audio/webm"
    if name.endswith(".m4a") or name.endswith(".mp4") or name.endswith(".aac"):
        return "audio/mp4"
    if name.endswith(".ogg"):
        return "audio/ogg"
    if name.endswith(".mp3"):
        return "audio/mpeg"
    return "application/octet-stream"


class Transcriber(ABC):
    """Turns a blob of recorded audio into text."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs (e.g. ``"elevenlabs-scribe"``)."""

    @abstractmethod
    async def transcribe(self, audio: bytes, *, filename: str = "audio.webm") -> str:
        """Return the transcript of ``audio`` (empty string if nothing heard)."""

    async def aclose(self) -> None:
        return None


class ElevenLabsTranscriber(Transcriber):
    """Speech-to-text via ElevenLabs Scribe. Reuses the existing ElevenLabs key."""

    def __init__(self, api_key: str, *, model_id: str = "scribe_v1", client: Any | None = None) -> None:
        if client is None:
            from elevenlabs.client import AsyncElevenLabs  # lazy: optional dep

            client = AsyncElevenLabs(api_key=api_key)
        self._client = client
        self._model_id = model_id

    @property
    def name(self) -> str:
        return f"elevenlabs-{self._model_id}"

    async def transcribe(self, audio: bytes, *, filename: str = "audio.wav") -> str:
        if not audio:
            return ""
        content_type = _content_type_for(filename)
        logger.info("STT upload %s (%d bytes, %s)", filename, len(audio), content_type)
        # Filename + content-type help Scribe detect the container. Prefer WAV
        # from the phone mic — Safari MediaRecorder blobs are often rejected.
        response = await self._client.speech_to_text.convert(
            model_id=self._model_id,
            file=(filename, audio, content_type),
        )
        text = getattr(response, "text", "") or ""
        return text.strip()

    async def aclose(self) -> None:
        closer = getattr(self._client, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception:  # noqa: BLE001
                pass
