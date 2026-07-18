"""A single conversation's structured memory."""

from __future__ import annotations

from app.models.message import Message, Role


class Conversation:
    """An append-only log of messages for one conversation.

    Provides the slices the orchestration core needs to assemble agent context:
    the full (bounded) history, and the "live discussion" — everything spoken
    since the most recent user message.

    This is deliberately a plain class (not a Pydantic model): it owns mutable
    state and behaviour, whereas :class:`~app.models.message.Message` is the
    immutable value it stores.
    """

    def __init__(self, conversation_id: str, *, max_history: int = 200) -> None:
        self._id = conversation_id
        self._messages: list[Message] = []
        self._max_history = max_history

    @property
    def id(self) -> str:
        return self._id

    @property
    def messages(self) -> list[Message]:
        """A copy of all retained messages (oldest first)."""
        return list(self._messages)

    def add(self, message: Message) -> Message:
        """Append a message, trimming to ``max_history`` if needed."""
        self._messages.append(message)
        if len(self._messages) > self._max_history:
            # Keep the most recent window; long-term recall is a future layer.
            self._messages = self._messages[-self._max_history :]
        return message

    def add_user(self, content: str) -> Message:
        return self.add(Message(role=Role.USER, speaker="User", content=content))

    def add_agent(self, speaker: str, content: str, *, intent: str | None = None, target: str | None = None) -> Message:
        return self.add(
            Message(role=Role.AGENT, speaker=speaker, content=content, intent=intent, target=target)
        )

    def recent(self, limit: int) -> list[Message]:
        """The last ``limit`` messages (oldest first)."""
        if limit <= 0:
            return []
        return list(self._messages[-limit:])

    def last_user_message(self) -> Message | None:
        for message in reversed(self._messages):
            if message.role is Role.USER:
                return message
        return None

    def prior_history(self) -> list[Message]:
        """All messages *before* the most recent user message (past turns)."""
        for index in range(len(self._messages) - 1, -1, -1):
            if self._messages[index].role is Role.USER:
                return list(self._messages[:index])
        return list(self._messages)

    def current_discussion(self) -> list[Message]:
        """Agent messages spoken since (and including nothing before) the last
        user message — i.e. the reply thread currently in flight."""
        discussion: list[Message] = []
        for message in reversed(self._messages):
            if message.role is Role.USER:
                break
            discussion.append(message)
        discussion.reverse()
        return discussion

    def transcript(self, limit: int | None = None) -> str:
        """A newline-joined ``Speaker: content`` transcript for prompts."""
        messages = self._messages if limit is None else self.recent(limit)
        return "\n".join(m.as_transcript_line() for m in messages)
