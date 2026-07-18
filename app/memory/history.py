"""A store of conversations — the seam for multi-session + long-term memory."""

from __future__ import annotations

from uuid import uuid4

from app.memory.conversation import Conversation


class History:
    """Owns every :class:`Conversation` in the process.

    Today it is an in-memory dict keyed by conversation id. The interface is
    kept deliberately small so a future implementation can persist to disk or a
    vector store (for long-term recall) and support many simultaneous
    conversations without changing callers.
    """

    def __init__(self, *, max_history: int = 200) -> None:
        self._conversations: dict[str, Conversation] = {}
        self._max_history = max_history

    def create(self, conversation_id: str | None = None) -> Conversation:
        conversation_id = conversation_id or uuid4().hex
        conversation = Conversation(conversation_id, max_history=self._max_history)
        self._conversations[conversation_id] = conversation
        return conversation

    def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def get_or_create(self, conversation_id: str) -> Conversation:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            conversation = self.create(conversation_id)
        return conversation

    def all(self) -> list[Conversation]:
        return list(self._conversations.values())
