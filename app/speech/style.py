"""Shared, presentational per-speaker colours.

Purely cosmetic and centralized here so the terminal player, the ElevenLabs
backend and the CLI never drift out of sync on who is what colour.
"""

from __future__ import annotations

_SPEAKER_COLORS: dict[str, str] = {
    "Gemini": "green",
    "Anthropic": "orange1",
    "DeepSeek": "blue",
    "Grok": "red",
    "Clerk": "bright_white",
}

# RGB equivalents of the same colours, for physical lighting (WiZ bulb, LEDs).
# Kept beside the terminal colours so a member is the *same* colour everywhere.
_SPEAKER_RGB: dict[str, tuple[int, int, int]] = {
    "Gemini": (40, 200, 90),
    "Anthropic": (255, 140, 0),
    "DeepSeek": (40, 120, 255),
    "Grok": (255, 40, 40),
    "Clerk": (220, 220, 230),
}

_DEFAULT_RGB = (255, 255, 255)


def color_for(name: str) -> str:
    """The base colour for a speaker (falls back to white)."""
    return _SPEAKER_COLORS.get(name, "white")


def style_for(name: str) -> str:
    """A bold Rich style string for a speaker's name."""
    return f"bold {color_for(name)}"


def rgb_for(name: str) -> tuple[int, int, int]:
    """The RGB colour for a speaker's physical light (falls back to white)."""
    return _SPEAKER_RGB.get(name, _DEFAULT_RGB)
