"""The full event vocabulary of the AI Council.

This module *is* the contract between the orchestration core and every
peripheral (speech, LEDs, logging, and future hardware). Peripherals subscribe
to these events; the core never imports peripherals. Adding a new device later
means adding a subscriber here — never editing business logic.
"""

from __future__ import annotations

from app.events.base import Event
from app.models.proposal import Intent, Proposal
from app.models.response import Response

# --------------------------------------------------------------------------- #
# Conversation lifecycle
# --------------------------------------------------------------------------- #


class UserMessage(Event):
    """A human turn entered the system."""

    conversation_id: str
    text: str


class ConversationEnded(Event):
    """A conversation was closed (session end, timeout, explicit stop)."""

    conversation_id: str
    reason: str = "ended"


# --------------------------------------------------------------------------- #
# Proposal phase
# --------------------------------------------------------------------------- #


class ProposalCreated(Event):
    """An agent returned its phase-1 proposal (bid or pass)."""

    conversation_id: str
    proposal: Proposal


class ProposalAccepted(Event):
    """The moderator scheduled a proposal to speak."""

    conversation_id: str
    proposal: Proposal
    position: int


class ProposalRejected(Event):
    """The moderator declined a proposal (silence, cap, fairness, low score)."""

    conversation_id: str
    proposal: Proposal
    reason: str


# --------------------------------------------------------------------------- #
# Scheduling / speaking queue
# --------------------------------------------------------------------------- #


class SpeechQueued(Event):
    """A speaker was placed into the speaking queue."""

    conversation_id: str
    speaker: str
    intent: Intent
    position: int
    is_interrupt: bool = False


class InterruptRequest(Event):
    """An agent asked to speak *after* the current speaker finishes.

    Agents never cut off live speech; they raise this and the moderator decides
    whether to slot them to the front of the queue.
    """

    conversation_id: str
    agent: str
    reason: str
    confidence: int
    intent: Intent
    target: str | None = None


# --------------------------------------------------------------------------- #
# Speech layer (the ONLY events peripherals like TTS / LEDs care about)
# --------------------------------------------------------------------------- #


class SpeakEvent(Event):
    """A request to voice a line. The council emits this; speech renders it.

    Nothing outside the speech layer knows how this becomes sound (or LEDs, or
    a terminal print). ``correlation_id`` lets a caller await the matching
    :class:`SpeechFinished`.
    """

    conversation_id: str
    speaker: str
    text: str
    intent: Intent
    correlation_id: str


class SpeechStarted(Event):
    """The speech layer began rendering a :class:`SpeakEvent`."""

    conversation_id: str
    speaker: str
    correlation_id: str


class SpeechProgress(Event):
    """Incremental progress while speaking — the LED/animation hook.

    ``amplitude`` is a normalized ``0.0..1.0`` loudness estimate. Today it is
    synthesized from text; with ElevenLabs it will come from real audio chunks.
    """

    conversation_id: str
    speaker: str
    correlation_id: str
    amplitude: float
    text_segment: str | None = None


class SpeechFinished(Event):
    """The speech layer finished rendering a :class:`SpeakEvent`."""

    conversation_id: str
    speaker: str
    correlation_id: str
