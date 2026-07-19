"""WiZ smart-bulb lighting backend (pywizlight over UDP).

Talks directly to a bulb by IP. The SDK is imported lazily so the dependency
stays optional — a machine without ``pywizlight`` (or without a bulb) simply
never constructs this backend, and the council runs unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from app.lighting.base import RGB, LightBackend

logger = logging.getLogger("ai_council.lighting.wiz")


class WizLightBackend(LightBackend):
    """A single WiZ bulb addressed by IP."""

    def __init__(self, ip: str, *, off_on_close: bool = True, light: Any | None = None) -> None:
        self._ip = ip
        self._off_on_close = off_on_close
        self._light = light  # lazily created on first use

    @property
    def name(self) -> str:
        return "wiz"

    def _bulb(self) -> Any:
        if self._light is None:
            from pywizlight import wizlight  # lazy: keeps the dep optional

            self._light = wizlight(self._ip)
        return self._light

    async def set_color(self, rgb: RGB, brightness: int) -> None:
        from pywizlight import PilotBuilder

        brightness = max(0, min(255, int(brightness)))
        await self._bulb().turn_on(PilotBuilder(rgb=rgb, brightness=brightness))

    async def off(self) -> None:
        await self._bulb().turn_off()

    async def aclose(self) -> None:
        if self._light is None:
            return
        try:
            if self._off_on_close:
                await self._light.turn_off()
        except Exception:  # noqa: BLE001 — best-effort on shutdown
            pass
        # Newer pywizlight exposes an explicit transport close; use it if present.
        closer = getattr(self._light, "async_close", None)
        if closer is not None:
            try:
                await closer()
            except Exception:  # noqa: BLE001
                pass
