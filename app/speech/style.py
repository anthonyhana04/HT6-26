"""Shared, presentational per-speaker colours.

Purely cosmetic and centralized here so the terminal player, the ElevenLabs
backend and the CLI never drift out of sync on who is what colour.
"""

from __future__ import annotations

_SPEAKER_COLORS: dict[str, str] = {
    "Gemini": "yellow",
    "Anthropic": "magenta",
    "DeepSeek": "cyan",
    "Groq": "red",
}


def color_for(name: str) -> str:
    """The base colour for a speaker (falls back to white)."""
    return _SPEAKER_COLORS.get(name, "white")


def style_for(name: str) -> str:
    """A bold Rich style string for a speaker's name."""
    return f"bold {color_for(name)}"
