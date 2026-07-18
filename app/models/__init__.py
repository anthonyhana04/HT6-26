"""Immutable data structures shared across the engine.

These types are deliberately free of behaviour and dependencies so they can be
passed across every layer (agents, council, speech, memory) and, eventually,
serialized across process/network boundaries for the Raspberry Pi + API future.
"""

from app.models.context import AgentContext, QueuedSpeechView
from app.models.message import Message, Role
from app.models.proposal import Intent, Proposal
from app.models.response import Response

__all__ = [
    "AgentContext",
    "QueuedSpeechView",
    "Message",
    "Role",
    "Intent",
    "Proposal",
    "Response",
]
