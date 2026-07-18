"""An in-process, asyncio-native publish/subscribe event bus.

The bus is the single integration seam of the whole system. The orchestration
core publishes events; agents, speech, memory, logging and (later) hardware
subscribe. Handlers may be sync or async. Subscription is keyed by event class,
so there is no string-topic bookkeeping.

Design notes
------------
* ``publish`` awaits all matching handlers, so callers can rely on an event
  being fully processed once ``await bus.publish(...)`` returns. Handlers run
  concurrently with :func:`asyncio.gather`.
* A handler raising an exception is logged and isolated — one bad subscriber
  never breaks a publish or other subscribers.
* :meth:`wait_for` turns the bus into a coordination primitive (used by the
  scheduler to await ``SpeechFinished`` without polling).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from typing import Awaitable, Callable, TypeVar

from app.events.base import Event

logger = logging.getLogger("ai_council.events")

E = TypeVar("E", bound=Event)

# A handler may return ``None`` (sync) or a coroutine (async).
Handler = Callable[[Event], Awaitable[None] | None]
Predicate = Callable[[Event], bool]


class Subscription:
    """A cancellable handle returned by :meth:`EventBus.subscribe`."""

    __slots__ = ("_bus", "_event_type", "_handler", "_active")

    def __init__(self, bus: "EventBus", event_type: type[Event] | None, handler: Handler) -> None:
        self._bus = bus
        self._event_type = event_type
        self._handler = handler
        self._active = True

    def cancel(self) -> None:
        """Remove this subscription from the bus (idempotent)."""
        if self._active:
            self._bus._remove(self._event_type, self._handler)
            self._active = False


class EventBus:
    """A minimal, dependency-free async event bus."""

    def __init__(self) -> None:
        # Exact-type subscribers.
        self._handlers: dict[type[Event], list[Handler]] = defaultdict(list)
        # Wildcard subscribers (receive every event).
        self._wildcard: list[Handler] = []

    # -- subscription -------------------------------------------------------- #

    def subscribe(self, event_type: type[E], handler: Callable[[E], Awaitable[None] | None]) -> Subscription:
        """Subscribe ``handler`` to a single event type."""
        self._handlers[event_type].append(handler)  # type: ignore[arg-type]
        return Subscription(self, event_type, handler)  # type: ignore[arg-type]

    def subscribe_all(self, handler: Handler) -> Subscription:
        """Subscribe ``handler`` to *every* event (useful for logging)."""
        self._wildcard.append(handler)
        return Subscription(self, None, handler)

    def _remove(self, event_type: type[Event] | None, handler: Handler) -> None:
        target = self._wildcard if event_type is None else self._handlers.get(event_type, [])
        try:
            target.remove(handler)
        except ValueError:
            pass

    # -- publishing ---------------------------------------------------------- #

    async def publish(self, event: Event) -> None:
        """Dispatch ``event`` to all matching handlers concurrently.

        Returns once every handler has completed. Exceptions are captured and
        logged so a faulty subscriber cannot poison the publish.
        """
        handlers = list(self._handlers.get(type(event), ())) + list(self._wildcard)
        if not handlers:
            return

        results = await asyncio.gather(
            *(self._invoke(handler, event) for handler in handlers),
            return_exceptions=True,
        )
        for handler, result in zip(handlers, results):
            if isinstance(result, Exception):
                logger.exception(
                    "Event handler %r failed for %s", getattr(handler, "__qualname__", handler), event.name,
                    exc_info=result,
                )

    @staticmethod
    async def _invoke(handler: Handler, event: Event) -> None:
        result = handler(event)
        if inspect.isawaitable(result):
            await result

    # -- coordination -------------------------------------------------------- #

    async def wait_for(
        self,
        event_type: type[E],
        predicate: Predicate | None = None,
        *,
        timeout: float | None = None,
    ) -> E:
        """Await the next event of ``event_type`` matching ``predicate``.

        This lets the orchestration core coordinate on events (e.g. "resume
        once *this* speech finishes") without busy-waiting.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[E] = loop.create_future()

        def _handler(event: Event) -> None:
            if future.done():
                return
            if predicate is None or predicate(event):  # type: ignore[arg-type]
                future.set_result(event)  # type: ignore[arg-type]

        subscription = self.subscribe(event_type, _handler)
        try:
            if timeout is not None:
                return await asyncio.wait_for(future, timeout)
            return await future
        finally:
            subscription.cancel()
