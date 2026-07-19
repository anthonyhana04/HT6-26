"""The speaking queue — a FIFO with a privileged front for interrupts."""

from __future__ import annotations

from collections import deque

from pydantic import BaseModel, ConfigDict

from app.models.context import QueuedSpeechView
from app.models.proposal import Proposal


class QueuedSpeech(BaseModel):
    """A scheduled speaking slot backed by the winning proposal."""

    model_config = ConfigDict(frozen=True)

    proposal: Proposal
    is_interrupt: bool = False

    @property
    def agent(self) -> str:
        return self.proposal.agent

    def to_view(self) -> QueuedSpeechView:
        return QueuedSpeechView(
            agent=self.proposal.agent,
            intent=self.proposal.intent.value,
            is_interrupt=self.is_interrupt,
        )


class SpeakingQueue:
    """Ordered speaking slots.

    Normal turns are appended (fair FIFO). Interrupts are pushed to the *front*
    so an accepted ``InterruptRequest`` speaks next — but only after the current
    speaker finishes, because the scheduler pops one slot at a time.
    """

    def __init__(self) -> None:
        self._items: deque[QueuedSpeech] = deque()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def is_empty(self) -> bool:
        return not self._items

    def enqueue(self, item: QueuedSpeech) -> int:
        """Append to the back. Returns the item's 0-based position."""
        self._items.append(item)
        return len(self._items) - 1

    def enqueue_front(self, item: QueuedSpeech) -> None:
        """Push to the front (interrupt priority)."""
        self._items.appendleft(item)

    def pop(self) -> QueuedSpeech | None:
        """Remove and return the next speaker, or ``None`` if empty."""
        return self._items.popleft() if self._items else None

    def peek(self) -> QueuedSpeech | None:
        """Next speaker without removing them, or ``None`` if empty."""
        return self._items[0] if self._items else None

    def snapshot(self) -> list[QueuedSpeechView]:
        """A read-only view of everyone still waiting (for agent context)."""
        return [item.to_view() for item in self._items]

    def clear(self) -> None:
        self._items.clear()
