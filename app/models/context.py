"""The read-only view of the world handed to an agent in either phase.

An :class:`AgentContext` is assembled by the council for each agent on each
turn. It intentionally contains everything an agent needs to make an autonomous
decision — history, live discussion, peers' responses and the speaking queue —
and nothing that would let it mutate shared state.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.models.message import Message
from app.models.proposal import Proposal
from app.models.response import Response


class Phase(str, Enum):
    """Which half of the two-phase cycle the context is being built for."""

    PROPOSAL = "proposal"
    GENERATION = "generation"


class QueuedSpeechView(BaseModel):
    """A read-only snapshot of one queued speaker, safe to show agents."""

    model_config = ConfigDict(frozen=True)

    agent: str
    intent: str
    is_interrupt: bool = False


class AgentContext(BaseModel):
    """Everything an agent sees when deciding whether/what to contribute."""

    model_config = ConfigDict(frozen=True)

    phase: Phase

    # The message that triggered this discussion cycle.
    user_message: Message

    # Full prior conversation (bounded by the memory layer).
    history: list[Message] = Field(default_factory=list)

    # Messages spoken since the triggering user message (the "live" discussion).
    discussion: list[Message] = Field(default_factory=list)

    # Responses produced by peers earlier in this very turn.
    peer_responses: list[Response] = Field(default_factory=list)

    # Who is currently waiting to speak.
    queue: list[QueuedSpeechView] = Field(default_factory=list)

    # In an interrupt round, the utterance the agent is reacting to.
    speaking_now: Message | None = None

    # During the generation phase, the intent the agent committed to in phase 1.
    committed_proposal: Proposal | None = None

    # True during the closing round, when the Lead may offer a final summary.
    is_closing: bool = False
