"""The lighting layer's bus adapter — turns speech events into light.

This is the LED lamp the architecture always promised: a pure *subscriber* to
the speech lifecycle. It listens to the same three events the terminal and
voices produce and drives a physical light:

    SpeechStarted   → bulb switches to the speaker's colour
    SpeechProgress  → brightness pulses with the voice amplitude
    SpeechFinished  → bulb turns off (idle)

Two design decisions keep it safe:

* **Handlers are synchronous and instant.** They only record the desired light
  state and wake a worker. They never ``await`` the (UDP, ~tens of ms) bulb call
  inside :meth:`EventBus.publish`, which would otherwise stall audio playback.
* **A coalescing worker** applies only the *latest* desired state. Rapid
  amplitude updates collapse into "converge to newest" instead of a backlog, so
  the bulb never lags behind the conversation.

The council core knows nothing about any of this.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.events.event_bus import EventBus
from app.events.event_types import SpeechFinished, SpeechProgress, SpeechStarted
from app.lighting.base import RGB, LightBackend

logger = logging.getLogger("ai_council.lighting")

# Amplitude (0..1) maps into this brightness band so the bulb always stays
# clearly lit while still visibly pulsing with the voice.
_MIN_BRIGHTNESS = 80
_MAX_BRIGHTNESS = 255


@dataclass(frozen=True)
class _LightState:
    on: bool
    rgb: RGB
    brightness: int


class LightService:
    """Drives a :class:`LightBackend` from the speech event stream."""

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
        self._last_brightness = -1

        self._subs = [
            bus.subscribe(SpeechStarted, self._on_started),
            bus.subscribe(SpeechProgress, self._on_progress),
            bus.subscribe(SpeechFinished, self._on_finished),
        ]

    @property
    def backend(self) -> LightBackend:
        return self._backend

    # -- event handlers (sync + instant: only set state, never await hw) ----- #

    def _on_started(self, event: SpeechStarted) -> None:
        rgb = self._colors.get(event.speaker, self._default_color)
        self._set_target(_LightState(on=True, rgb=rgb, brightness=_MAX_BRIGHTNESS))

    def _on_progress(self, event: SpeechProgress) -> None:
        rgb = self._colors.get(event.speaker, self._default_color)
        brightness = _amplitude_to_brightness(event.amplitude)
        # Skip micro-changes to avoid needless bulb chatter / flicker.
        if abs(brightness - self._last_brightness) < 12:
            return
        self._last_brightness = brightness
        self._set_target(_LightState(on=True, rgb=rgb, brightness=brightness))

    def _on_finished(self, event: SpeechFinished) -> None:
        self._last_brightness = -1
        self._set_target(_LightState(on=False, rgb=self._default_color, brightness=0))

    def _set_target(self, state: _LightState) -> None:
        self._target = state
        self._dirty.set()
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        if self._worker is None or self._worker.done():
            try:
                self._worker = asyncio.get_running_loop().create_task(self._run())
            except RuntimeError:
                # No running loop (shouldn't happen from a bus handler); ignore.
                pass

    # -- coalescing worker: applies only the newest desired state ------------ #

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
            except Exception:  # noqa: BLE001 — hardware is best-effort, never fatal
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
