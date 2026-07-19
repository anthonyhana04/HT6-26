"""The lighting backend abstraction.

A :class:`LightBackend` is the physical-light analogue of a speech backend: it
turns a desired colour/brightness into hardware state. Keeping it behind an ABC
means the :class:`~app.lighting.service.LightService` (which listens to the bus)
never knows whether it's driving a WiZ bulb, an LED strip, or nothing at all.
Swapping hardware later is a one-line change in the composition root.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

RGB = tuple[int, int, int]


class LightBackend(ABC):
    """Drives a single physical light. Implementations must tolerate repeated,
    rapid calls and must never raise into the caller (best-effort hardware)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs (e.g. ``"wiz"``)."""

    @abstractmethod
    async def set_color(self, rgb: RGB, brightness: int) -> None:
        """Turn the light on at ``rgb`` with ``brightness`` in ``0..255``."""

    @abstractmethod
    async def off(self) -> None:
        """Turn the light off."""

    async def aclose(self) -> None:
        """Release any resources (sockets). Default: no-op."""
        return None
