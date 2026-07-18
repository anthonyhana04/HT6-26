"""The terminal speech backend — the CLI milestone's "speaker".

It prints each council member's line with Rich styling and yields a synthetic
amplitude envelope so the ``SpeechProgress`` event stream (and therefore any
LED subscriber) behaves exactly as it will with real audio.
"""

from __future__ import annotations

import asyncio
import re
from typing import AsyncIterator

from rich.console import Console

from app.events.event_types import SpeakEvent
from app.speech.base import SpeechBackend, SpeechChunk
from app.speech.style import style_for

_WORD_RE = re.compile(r"\S+\s*")


class TerminalPlayer(SpeechBackend):
    """Speech backend that "speaks" by printing to a Rich console."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        words_per_minute: float = 420.0,
        animate: bool = True,
    ) -> None:
        self._console = console or Console()
        # Delay between word chunks, derived from a reading pace.
        self._word_delay = 60.0 / max(words_per_minute, 1.0)
        self._animate = animate

    @property
    def name(self) -> str:
        return "terminal"

    async def stream(self, event: SpeakEvent) -> AsyncIterator[SpeechChunk]:
        text = event.text.strip()
        style = style_for(event.speaker)

        label = f"[{style}]{event.speaker}[/]"
        intent = f"[dim]({event.intent.value.lower()})[/dim]"
        self._console.print(f"{label} {intent}", end=" ")

        if not text:
            self._console.print()
            return

        for match in _WORD_RE.finditer(text):
            word = match.group(0)
            self._console.print(word, end="", style=style, highlight=False, soft_wrap=True)
            yield SpeechChunk(amplitude=_amplitude_for(word), text_segment=word)
            if self._animate and self._word_delay > 0:
                await asyncio.sleep(self._word_delay)

        self._console.print()  # newline terminates the utterance


def _amplitude_for(word: str) -> float:
    """A crude but stable loudness proxy from a word's shape.

    Longer words and emphatic punctuation read "louder". This is what the LED
    lamp will consume until real audio envelopes replace it.
    """
    stripped = word.strip()
    if not stripped:
        return 0.0
    base = min(len(stripped) / 10.0, 1.0)
    if stripped.endswith(("!", "?")):
        base = min(base + 0.25, 1.0)
    return round(max(base, 0.15), 3)
