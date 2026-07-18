"""The scheduler: drives a single user message through the whole lifecycle.

    UserMessage
        → proposal round (all agents, in parallel)          [phase 1]
        → moderator selects + orders speakers
        → speaking queue built
        → for each speaker:
              generate full response                        [phase 2]
              emit SpeakEvent, await SpeechFinished
              run an interrupt round about what was said
                  accepted interrupts jump the queue
        → stop when the queue drains or the turn cap is hit

The scheduler never renders speech itself and never touches an LLM SDK: it only
asks agents to think, asks the moderator to decide, and emits events.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from app.agents.base import BaseAgent
from app.council.moderator import Moderator
from app.council.queue import QueuedSpeech, SpeakingQueue
from app.events.event_bus import EventBus
from app.events.event_types import ProposalCreated, SpeakEvent, SpeechFinished, SpeechQueued
from app.memory.conversation import Conversation
from app.models.context import AgentContext, Phase
from app.models.message import Message
from app.models.proposal import Proposal
from app.models.response import Response

logger = logging.getLogger("ai_council.scheduler")


class Scheduler:
    """Orchestrates proposal → generation → speech → interrupts for one turn."""

    def __init__(
        self,
        bus: EventBus,
        moderator: Moderator,
        agents: list[BaseAgent],
        *,
        max_turns: int = 8,
    ) -> None:
        self._bus = bus
        self._moderator = moderator
        self._agents = list(agents)
        self._agents_by_name = {agent.name: agent for agent in agents}
        self._lead = next((agent for agent in agents if agent.is_lead), None)
        self._max_turns = max_turns

    async def run_turn(self, conversation: Conversation) -> None:
        """Process the most recent user message end to end."""
        user_message = conversation.last_user_message()
        if user_message is None:
            return
        conversation_id = conversation.id

        self._moderator.reset_turn()
        queue = SpeakingQueue()
        peer_responses: list[Response] = []

        # --- Phase 1: proposal round over every member ---------------------- #
        proposals = await self._moderator.request_proposals(
            self._agents,
            lambda agent: self._build_context(
                conversation, queue, peer_responses, Phase.PROPOSAL
            ),
            conversation_id,
        )
        selection = await self._moderator.choose_speakers(proposals, conversation_id)
        for position, proposal in enumerate(selection.accepted):
            queue.enqueue(QueuedSpeech(proposal=proposal))
            await self._bus.publish(
                SpeechQueued(
                    conversation_id=conversation_id,
                    speaker=proposal.agent,
                    intent=proposal.intent,
                    position=position,
                )
            )

        # --- Drain the queue: generate, speak, then allow interrupts -------- #
        turns = 0
        while not queue.is_empty and turns < self._max_turns:
            item = queue.pop()
            assert item is not None
            turns += 1

            response = await self._generate(conversation, queue, peer_responses, item.proposal)
            spoken = conversation.add_agent(
                response.agent,
                response.text,
                intent=response.intent.value,
                target=response.target,
            )
            peer_responses.append(response)

            await self._speak(conversation_id, response)

            if turns < self._max_turns:
                await self._interrupt_round(conversation, queue, peer_responses, spoken)

        # --- Closing round: the Lead may take the last word ----------------- #
        await self._closing_round(conversation, queue, peer_responses)

    # -- phase 2 ------------------------------------------------------------- #

    async def _generate(
        self,
        conversation: Conversation,
        queue: SpeakingQueue,
        peer_responses: list[Response],
        proposal: Proposal,
    ) -> Response:
        agent = self._agents_by_name[proposal.agent]
        context = self._build_context(
            conversation, queue, peer_responses, Phase.GENERATION, committed=proposal
        )
        return await agent.generate(context)

    # -- speech (emit + await completion; guarantees no overlap) ------------- #

    async def _speak(self, conversation_id: str, response: Response) -> None:
        correlation_id = uuid4().hex
        speak = SpeakEvent(
            conversation_id=conversation_id,
            speaker=response.agent,
            text=response.text,
            intent=response.intent,
            correlation_id=correlation_id,
        )

        # Subscribe *before* publishing to avoid missing a synchronous finish.
        loop = asyncio.get_running_loop()
        finished: asyncio.Future[None] = loop.create_future()

        def _on_finished(event: SpeechFinished) -> None:
            if not finished.done() and event.correlation_id == correlation_id:
                finished.set_result(None)

        subscription = self._bus.subscribe(SpeechFinished, _on_finished)
        try:
            await self._bus.publish(speak)
            await finished
        finally:
            subscription.cancel()

    # -- interrupts ---------------------------------------------------------- #

    async def _interrupt_round(
        self,
        conversation: Conversation,
        queue: SpeakingQueue,
        peer_responses: list[Response],
        spoken: Message,
    ) -> None:
        """Give every member a chance to raise an interrupt about ``spoken``."""
        conversation_id = conversation.id
        proposals = await self._moderator.request_proposals(
            self._agents,
            lambda agent: self._build_context(
                conversation, queue, peer_responses, Phase.PROPOSAL, speaking_now=spoken
            ),
            conversation_id,
        )
        selection = await self._moderator.choose_speakers(
            proposals, conversation_id, interrupt=True
        )
        for proposal in selection.accepted:
            queue.enqueue_front(QueuedSpeech(proposal=proposal, is_interrupt=True))
            await self._bus.publish(
                SpeechQueued(
                    conversation_id=conversation_id,
                    speaker=proposal.agent,
                    intent=proposal.intent,
                    position=0,
                    is_interrupt=True,
                )
            )

    async def _closing_round(
        self,
        conversation: Conversation,
        queue: SpeakingQueue,
        peer_responses: list[Response],
    ) -> None:
        """Offer the Lead the final word when several members participated.

        This is proposal-driven, not forced: the Lead decides (and the moderator
        still applies fairness + threshold), so a summary only happens when it
        genuinely adds value.
        """
        if self._lead is None or len({r.agent for r in peer_responses}) < 2:
            return
        conversation_id = conversation.id

        context = self._build_context(
            conversation, queue, peer_responses, Phase.PROPOSAL, is_closing=True
        )
        proposal = await self._lead.propose(context)
        await self._bus.publish(ProposalCreated(conversation_id=conversation_id, proposal=proposal))

        selection = await self._moderator.choose_speakers([proposal], conversation_id)
        for accepted in selection.accepted:
            response = await self._generate(conversation, queue, peer_responses, accepted)
            conversation.add_agent(
                response.agent, response.text, intent=response.intent.value, target=response.target
            )
            peer_responses.append(response)
            await self._speak(conversation_id, response)

    # -- context assembly ---------------------------------------------------- #

    def _build_context(
        self,
        conversation: Conversation,
        queue: SpeakingQueue,
        peer_responses: list[Response],
        phase: Phase,
        *,
        committed: Proposal | None = None,
        speaking_now: Message | None = None,
        is_closing: bool = False,
    ) -> AgentContext:
        user_message = conversation.last_user_message()
        assert user_message is not None
        return AgentContext(
            phase=phase,
            user_message=user_message,
            history=conversation.prior_history(),
            discussion=conversation.current_discussion(),
            peer_responses=list(peer_responses),
            queue=queue.snapshot(),
            speaking_now=speaking_now,
            committed_proposal=committed,
            is_closing=is_closing,
        )
