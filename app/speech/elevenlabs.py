"""Streaming ElevenLabs speech backend — the council's real voice.

Contract reminder: nothing outside the speech layer knows this exists. The
council only emits ``SpeakEvent``; this backend renders it. It:

    SpeakEvent
      → ElevenLabs streaming TTS (per-speaker voice_id from configuration)
      → PCM audio chunks
      → played through the speaker (sounddevice) in real time
      → yielded as SpeechChunk(audio=..., amplitude=<real RMS>)
      → SpeechService turns those into SpeechProgress events (the LED hook)

Each council member gets a distinct voice. Voice IDs come from configuration;
any member without a configured voice is auto-assigned a distinct one from the
account's available voices, so unique voices work out of the box.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from rich.console import Console

from app.events.event_types import SpeakEvent
from app.speech.base import SpeechBackend, SpeechChunk
from app.speech.style import style_for

logger = logging.getLogger("ai_council.speech.elevenlabs")

# PCM sample rate we request from ElevenLabs and play back at.
_SAMPLE_RATE = 24_000
_OUTPUT_FORMAT = "pcm_24000"


class ElevenLabsSpeech(SpeechBackend):
    """Streaming text-to-speech with a distinct voice per council member."""

    def __init__(
        self,
        api_key: str,
        voice_ids: dict[str, str | None],
        *,
        model_id: str = "eleven_flash_v2_5",
        console: Console | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from elevenlabs.client import AsyncElevenLabs  # lazy: keeps SDK optional

            client = AsyncElevenLabs(api_key=api_key)
        self._client = client
        self._model_id = model_id
        self._console = console or Console()
        # name -> voice_id (values may be None until auto-assigned).
        self._voice_ids: dict[str, str | None] = dict(voice_ids)
        self._voices_ready = False
        self._audio_ok = True  # flips off if no output device is available

    @property
    def name(self) -> str:
        return "elevenlabs"

    def voice_for(self, speaker: str) -> str | None:
        return self._voice_ids.get(speaker)

    async def stream(self, event: SpeakEvent) -> AsyncIterator[SpeechChunk]:
        self._print_header(event)
        text = event.text.strip()
        if not text:
            return

        await self._ensure_voice_map()
        voice_id = self._voice_ids.get(event.speaker)
        if not voice_id:
            # No voice available for this speaker — degrade to text only.
            logger.warning("No ElevenLabs voice for %s; printing without audio", event.speaker)
            return

        player = _AudioPlayer(_SAMPLE_RATE) if self._audio_ok else None
        if player is not None and not player.ok:
            self._audio_ok = False
            player = None

        try:
            audio_stream = self._client.text_to_speech.stream(
                voice_id,
                text=text,
                model_id=self._model_id,
                output_format=_OUTPUT_FORMAT,
            )
            async for chunk in audio_stream:
                if not chunk:
                    continue
                if player is not None:
                    await player.write(chunk)
                yield SpeechChunk(amplitude=_rms_amplitude(chunk), audio=chunk, text_segment=None)
        finally:
            if player is not None:
                await player.close()

    async def _ensure_voice_map(self) -> None:
        """Fill in a distinct voice for any member that lacks one."""
        if self._voices_ready:
            return
        self._voices_ready = True

        missing = [name for name, vid in self._voice_ids.items() if not vid]
        if not missing:
            return
        try:
            response = await self._client.voices.get_all()
            used = {vid for vid in self._voice_ids.values() if vid}
            pool = [v.voice_id for v in response.voices if v.voice_id not in used]
            for name, voice_id in zip(missing, pool):
                self._voice_ids[name] = voice_id
                logger.info("Auto-assigned ElevenLabs voice %s to %s", voice_id, name)
        except Exception:  # noqa: BLE001 — voice listing is best-effort
            logger.warning("Could not auto-assign ElevenLabs voices", exc_info=True)

    def _print_header(self, event: SpeakEvent) -> None:
        style = style_for(event.speaker)
        self._console.print(
            f"[{style}]{event.speaker}[/] [dim]({event.intent.value.lower()})[/dim] {event.text.strip()}"
        )

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001
                pass


class _AudioPlayer:
    """Thin real-time PCM player over sounddevice; no-ops if audio is absent."""

    def __init__(self, sample_rate: int) -> None:
        self._stream = None
        self.ok = False
        try:
            import sounddevice as sd

            self._stream = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16")
            self._stream.start()
            self.ok = True
        except Exception:  # noqa: BLE001 — headless/CI or no PortAudio device
            logger.warning("Audio output unavailable; speaking silently", exc_info=True)

    async def write(self, chunk: bytes) -> None:
        if self._stream is None:
            return
        # Blocking write paces playback to real time; keep the loop responsive.
        await asyncio.to_thread(self._stream.write, chunk)

    async def close(self) -> None:
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        try:
            await asyncio.to_thread(stream.stop)
            stream.close()
        except Exception:  # noqa: BLE001
            pass


def _rms_amplitude(chunk: bytes) -> float:
    """Normalized 0..1 loudness of a 16-bit PCM chunk (the LED signal)."""
    try:
        import numpy as np

        samples = np.frombuffer(chunk, dtype=np.int16)
        if samples.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
        # Speech RMS is small; scale into a visible range and clamp.
        return round(min(rms * 3.5, 1.0), 3)
    except Exception:  # noqa: BLE001 — never let metering break playback
        return 0.0
