"""The lighting layer's bus adapter — turns speech + listening events into light.

A pure *subscriber* to the event bus:

    ListeningStateChanged(awaiting_command) → solid white (user interrupted, recording)
    SpeechStarted   → bulb switches to the speaker's colour
    SpeechProgress  → brightness pulses with the voice amplitude
    SpeechFinished  → bulb turns off (unless still awaiting a command)

Handlers stay synchronous and instant; a coalescing worker applies hardware
updates so the audio path is never stalled by UDP bulb calls.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.events.event_bus import EventBus
from app.events.event_types import (
    ListeningStateChanged,
    SpeechFinished,
    SpeechProgress,
    SpeechStarted,
)
from app.lighting.base import RGB, LightBackend

logger = logging.getLogger("ai_council.lighting")

_MIN_BRIGHTNESS = 80
_MAX_BRIGHTNESS = 255
_WAKE_WHITE: RGB = (255, 255, 255)


@dataclass(frozen=True)
class _LightState:
    on: bool
    rgb: RGB
    brightness: int


class LightService:
    """Drives a :class:`LightBackend` from speech + listening events."""

    def __init__(
        self,
        bus: EventBus,
        backend: LightBackend,
        colors: dict[str, RGB],
        *,
        default_color: RGB = (255, 255, 255),
    ) -> None:
        self._bus = bus
        self._backend = backend
        self._colors = dict(colors)
        self._default_color = default_color

        self._target: _LightState | None = None
        self._dirty = asyncio.Event()
        self._worker: asyncio.Task[None] | None = None
        self._listening_state = "idle"
        self._last_brightness = -1

        self._subs = [
            bus.subscribe(ListeningStateChanged, self._on_listening),
            bus.subscribe(SpeechStarted, self._on_started),
            bus.subscribe(SpeechProgress, self._on_progress),
            bus.subscribe(SpeechFinished, self._on_finished),
        ]

    @property
    def backend(self) -> LightBackend:
        return self._backend

    # -- event handlers (sync + instant) ------------------------------------- #

    def _on_listening(self, event: ListeningStateChanged) -> None:
        self._listening_state = event.state
        if event.state == "awaiting_command":
            # User interrupted — solid white while they finish their clip.
            self._set_target(_LightState(on=True, rgb=_WAKE_WHITE, brightness=_MAX_BRIGHTNESS))
        elif event.state == "idle":
            self._last_brightness = -1
            self._set_target(_LightState(on=False, rgb=self._default_color, brightness=0))

    def _on_started(self, event: SpeechStarted) -> None:
        rgb = self._colors.get(event.speaker, self._default_color)
        self._set_target(_LightState(on=True, rgb=rgb, brightness=_MAX_BRIGHTNESS))

    def _on_progress(self, event: SpeechProgress) -> None:
        rgb = self._colors.get(event.speaker, self._default_color)
        brightness = _amplitude_to_brightness(event.amplitude)
        if abs(brightness - self._last_brightness) < 12:
            return
        self._last_brightness = brightness
        self._set_target(_LightState(on=True, rgb=rgb, brightness=brightness))

    def _on_finished(self, event: SpeechFinished) -> None:
        self._last_brightness = -1
        # Between speakers the lamp goes dark briefly. If we're still waiting
        # for a question after a wake, keep the white "listening" light on.
        if self._listening_state == "awaiting_command":
            self._set_target(_LightState(on=True, rgb=_WAKE_WHITE, brightness=_MAX_BRIGHTNESS))
            return
        self._set_target(_LightState(on=False, rgb=self._default_color, brightness=0))

    # -- coalescing worker --------------------------------------------------- #

    def _set_target(self, state: _LightState) -> None:
        self._target = state
        self._dirty.set()
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            try:
                self._worker = asyncio.get_running_loop().create_task(self._run())
            except RuntimeError:
                pass

    async def _run(self) -> None:
        while True:
            await self._dirty.wait()
            self._dirty.clear()
            state = self._target
            if state is None:
                continue
            try:
                if state.on:
                    await self._backend.set_color(state.rgb, state.brightness)
                else:
                    await self._backend.off()
            except Exception:  # noqa: BLE001
                logger.debug("Light update failed", exc_info=True)

    async def aclose(self) -> None:
        for sub in self._subs:
            sub.cancel()
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await self._backend.aclose()


def _amplitude_to_brightness(amplitude: float) -> int:
    amplitude = max(0.0, min(1.0, amplitude))
    return int(_MIN_BRIGHTNESS + amplitude * (_MAX_BRIGHTNESS - _MIN_BRIGHTNESS))
