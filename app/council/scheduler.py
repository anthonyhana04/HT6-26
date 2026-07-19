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
import random
from uuid import uuid4

from app.agents.base import BaseAgent
from app.config.profiles import CLERK
from app.council.moderator import Moderator
from app.council.queue import QueuedSpeech, SpeakingQueue
from app.events.event_bus import EventBus
from app.events.event_types import ProposalCreated, SpeakEvent, SpeechFinished, SpeechQueued
from app.memory.conversation import Conversation
from app.models.context import AgentContext, Phase
from app.models.message import Message
from app.models.proposal import Intent, Proposal
from app.models.response import Response

_ADJOURN_LINE = "Council adjourned."

logger = logging.getLogger("ai_council.scheduler")


def _same_slot(a: Proposal, b: Proposal) -> bool:
    """True when a prefetched response belongs to this queue slot."""
    return a.agent == b.agent and a.intent == b.intent and a.reason == b.reason


# Peer intents that can end the turn without a Lead wrap-up.
_SETTLING_INTENTS = frozenset(
    {
        Intent.ANSWER,
        Intent.SUMMARY,
        Intent.DISAGREEMENT,
        Intent.CORRECTION,
    }
)


def _settles_question(response: Response) -> bool:
    """True when this line is enough — no Gemini encore required."""
    return response.intent in _SETTLING_INTENTS


def _agent_spoke(responses: list[Response], name: str) -> bool:
    return any(r.agent == name for r in responses)


class Scheduler:
    """Orchestrates proposal → generation → speech → interrupts for one turn."""

    def __init__(
        self,
        bus: EventBus,
        moderator: Moderator,
        agents: list[BaseAgent],
        *,
        max_turns: int = 5,
        max_interrupts: int = 1,
    ) -> None:
        self._bus = bus
        self._moderator = moderator
        self._agents = list(agents)
        self._agents_by_name = {agent.name: agent for agent in agents}
        self._lead = next((agent for agent in agents if agent.is_lead), None)
        self._max_turns = max_turns
        self._max_interrupts = max_interrupts
        self._cancelled = False
        self._interrupts_used = 0

    def cancel(self) -> None:
        """Abort the in-flight turn after the current speak finishes (or is cut)."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    async def run_turn(self, conversation: Conversation) -> None:
        """Process the most recent user message end to end."""
        user_message = conversation.last_user_message()
        if user_message is None:
            return
        conversation_id = conversation.id

        self._cancelled = False
        self._interrupts_used = 0
        self._moderator.reset_turn()
        queue = SpeakingQueue()
        peer_responses: list[Response] = []

        # --- Phase 1: Lead answers immediately; peers bid in the background -- #
        turns = 0
        peers_since_lead = 0
        lead_name = self._lead.name if self._lead else None
        ready_response: Response | None = None
        ready_for: Proposal | None = None
        # When set, first Lead speak resolves who follows (during TTS).
        peers_opening_task: asyncio.Task[list[Proposal]] | None = None

        opened = await self._begin_opening(conversation, queue, peer_responses)
        if self._cancelled or opened is None:
            return
        ready_for, ready_response, peers_opening_task = opened

        # --- Drain the queue: generate, speak, then lead-aware follow-ups --- #
        # Overlap the *next* speaker's LLM work with the current audio so voices
        # hand off immediately instead of waiting for a full generate after TTS.

        while not queue.is_empty and turns < self._max_turns and not self._cancelled:
            item = queue.pop()
            assert item is not None
            turns += 1

            if (
                ready_response is not None
                and ready_for is not None
                and _same_slot(ready_for, item.proposal)
            ):
                response = ready_response
            else:
                response = await self._generate(
                    conversation, queue, peer_responses, item.proposal
                )
            ready_response = None
            ready_for = None

            if self._cancelled:
                if peers_opening_task is not None:
                    peers_opening_task.cancel()
                break
            spoken = conversation.add_agent(
                response.agent,
                response.text,
                intent=response.intent.value,
                target=response.target,
            )
            peer_responses.append(response)

            # Kick off whatever we already know comes next *during* playback.
            opening_peers = peers_opening_task
            peers_opening_task = None
            overlap = self._start_overlap(
                conversation,
                queue,
                peer_responses,
                spoken,
                response,
                lead_name=lead_name,
                peers_since_lead=peers_since_lead,
                peers_opening_task=opening_peers,
            )

            await self._speak(conversation_id, response)

            if self._cancelled:
                if overlap is not None:
                    overlap.cancel()
                break

            if overlap is not None:
                try:
                    ready_for, ready_response, lead_handled = await overlap
                except asyncio.CancelledError:
                    ready_for, ready_response, lead_handled = None, None, False
                except Exception:  # noqa: BLE001 — fall back to serial path
                    logger.exception("Overlap prep failed; continuing without prefetch")
                    ready_for, ready_response, lead_handled = None, None, False
            else:
                lead_handled = False

            if lead_name and response.agent == lead_name:
                peers_since_lead = 0
            else:
                peers_since_lead += 1

            if turns >= self._max_turns:
                break

            brutalist = self._moderator.config.brutalist_name

            # After the designated Lead (Gemini): allow queued peer / interrupt.
            if lead_name and response.agent == lead_name:
                if (
                    self._interrupts_used < self._max_interrupts
                    and queue.is_empty
                    and ready_response is None
                ):
                    await self._interrupt_round(conversation, queue, peer_responses, spoken)
                continue

            # After anyone else: only invite Gemini back if the question is still open.
            # Never end the turn before Grok has taken their guaranteed beat.
            if response.agent != lead_name:
                grok_done = _agent_spoke(peer_responses, brutalist)
                if _settles_question(response) and grok_done:
                    continue
                if not grok_done:
                    # Opener finished; Grok should already be queued via opening overlap.
                    continue
                if not lead_handled:
                    lead_queued = await self._offer_lead_reply(
                        conversation, queue, peer_responses, spoken
                    )
                else:
                    lead_queued = bool(
                        ready_response is not None
                        or any(i.agent == lead_name for i in queue.snapshot())
                    )
                if (
                    not lead_queued
                    and self._interrupts_used < self._max_interrupts
                ):
                    await self._interrupt_round(
                        conversation,
                        queue,
                        peer_responses,
                        spoken,
                        min_confidence_override=85,
                    )
                continue

        # --- Guarantee Grok, then optional Gemini wrap-up ------------------- #
        if not self._cancelled:
            await self._ensure_brutalist_spoke(conversation, queue, peer_responses)
            await self._closing_round(conversation, queue, peer_responses)

        # Neutral clerk closes the sitting (skipped on barge-in).
        if not self._cancelled:
            await self._announce_adjourned(conversation_id)

    # -- opening (time-to-first-speech) -------------------------------------- #

    def _pick_opener(self) -> BaseAgent:
        """Random opener among Gemini / Anthropic / DeepSeek — never Grok."""
        brutalist = self._moderator.config.brutalist_name
        pool = [a for a in self._agents if a.name != brutalist]
        if not pool:
            return self._agents[0]
        opener = random.choice(pool)
        logger.info("This turn opens with %s", opener.name)
        return opener

    async def _begin_opening(
        self,
        conversation: Conversation,
        queue: SpeakingQueue,
        peer_responses: list[Response],
    ) -> tuple[Proposal, Response, asyncio.Task[list[Proposal]] | None] | None:
        """Start a random chair's answer ASAP; peer proposals run in the background.

        Opener is randomly Gemini, Anthropic, or DeepSeek. Grok is reserved for
        a guaranteed follow-up beat (resolved during the opener's TTS).

        Returns ``(opener_proposal, opener_response, peers_task)`` or ``None``.
        """
        conversation_id = conversation.id
        build = lambda agent: self._build_context(
            conversation, queue, peer_responses, Phase.PROPOSAL
        )

        opener = self._pick_opener()
        opener_proposal = Proposal(
            agent=opener.name,
            should_speak=True,
            confidence=92,
            intent=Intent.ANSWER,
            reason=f"{opener.name} opens the floor.",
        )
        await self._bus.publish(
            ProposalCreated(conversation_id=conversation_id, proposal=opener_proposal)
        )

        peers = [a for a in self._agents if a.name != opener.name]
        peers_task: asyncio.Task[list[Proposal]] = asyncio.create_task(
            self._moderator.request_proposals(peers, build, conversation_id)
        )
        try:
            opener_response = await self._generate(
                conversation, queue, peer_responses, opener_proposal
            )
        except Exception:
            peers_task.cancel()
            raise

        if self._cancelled:
            peers_task.cancel()
            return None

        selection = await self._moderator.choose_speakers([opener_proposal], conversation_id)
        if not selection.accepted:
            peers_task.cancel()
            return None

        queue.enqueue(QueuedSpeech(proposal=opener_proposal))
        await self._bus.publish(
            SpeechQueued(
                conversation_id=conversation_id,
                speaker=opener_proposal.agent,
                intent=opener_proposal.intent,
                position=0,
            )
        )
        return opener_proposal, opener_response, peers_task

    async def _queue_opening_peer(
        self,
        conversation: Conversation,
        queue: SpeakingQueue,
        peer_responses: list[Response],
        peer_proposals: list[Proposal],
    ) -> tuple[Proposal | None, Response | None]:
        """Always queue Grok next (guaranteed beat), prefetching their line."""
        if self._cancelled:
            return None, None
        conversation_id = conversation.id
        brutalist = self._moderator.config.brutalist_name

        grok_bid = next(
            (p for p in peer_proposals if p.agent == brutalist and p.should_speak),
            None,
        )
        if grok_bid is None:
            grok_bid = Proposal(
                agent=brutalist,
                should_speak=True,
                confidence=88,
                intent=Intent.DISAGREEMENT,
                reason="Brutalist seat — always takes a turn.",
            )
            await self._bus.publish(
                ProposalCreated(conversation_id=conversation_id, proposal=grok_bid)
            )

        selection = await self._moderator.choose_speakers([grok_bid], conversation_id)
        # Guarantee holds even if the confidence bar would have filtered them.
        peer = selection.accepted[0] if selection.accepted else grok_bid
        queue.enqueue(QueuedSpeech(proposal=peer))
        await self._bus.publish(
            SpeechQueued(
                conversation_id=conversation_id,
                speaker=peer.agent,
                intent=peer.intent,
                position=0,
            )
        )
        ready = await self._generate(conversation, queue, peer_responses, peer)
        return peer, ready

    async def _ensure_brutalist_spoke(
        self,
        conversation: Conversation,
        queue: SpeakingQueue,
        peer_responses: list[Response],
    ) -> None:
        """Safety net: if Grok never got in, give them a forced beat now."""
        brutalist = self._moderator.config.brutalist_name
        if self._cancelled or _agent_spoke(peer_responses, brutalist):
            return
        if brutalist not in self._agents_by_name:
            return
        proposal = Proposal(
            agent=brutalist,
            should_speak=True,
            confidence=90,
            intent=Intent.DISAGREEMENT,
            reason="Brutalist seat — guaranteed turn.",
        )
        await self._bus.publish(
            ProposalCreated(conversation_id=conversation.id, proposal=proposal)
        )
        response = await self._generate(conversation, queue, peer_responses, proposal)
        conversation.add_agent(
            response.agent,
            response.text,
            intent=response.intent.value,
            target=response.target,
        )
        peer_responses.append(response)
        await self._speak(conversation.id, response)

    # -- phase 2 ------------------------------------------------------------- #

    def _start_overlap(
        self,
        conversation: Conversation,
        queue: SpeakingQueue,
        peer_responses: list[Response],
        spoken: Message,
        response: Response,
        *,
        lead_name: str | None,
        peers_since_lead: int,
        peers_opening_task: asyncio.Task[list[Proposal]] | None = None,
    ) -> asyncio.Task[tuple[Proposal | None, Response | None, bool]] | None:
        """While ``response`` plays, prepare the next speaker's text if known.

        Returns a task of ``(proposal, response, lead_handled)``.
        """

        async def _run_opening_peers() -> tuple[Proposal | None, Response | None, bool]:
            assert peers_opening_task is not None
            try:
                peer_proposals = await peers_opening_task
            except asyncio.CancelledError:
                return None, None, False
            prop, ready = await self._queue_opening_peer(
                conversation, queue, peer_responses, peer_proposals
            )
            return prop, ready, False

        async def _run() -> tuple[Proposal | None, Response | None, bool]:
            # Peer just spoke → maybe prepare a Lead reply (skip if peer settled it).
            next_peers = peers_since_lead + (0 if response.agent == lead_name else 1)
            if lead_name and response.agent != lead_name and next_peers >= 1:
                if _settles_question(response):
                    return None, None, True
                queued = await self._offer_lead_reply(
                    conversation, queue, peer_responses, spoken
                )
                if not queued:
                    return None, None, True
                nxt = queue.peek()
                if nxt is None:
                    return None, None, True
                ready = await self._generate(
                    conversation, queue, peer_responses, nxt.proposal
                )
                return nxt.proposal, ready, True

            # Lead (or open floor) with someone already queued → prefetch them.
            nxt = queue.peek()
            if nxt is None:
                return None, None, False
            ready = await self._generate(conversation, queue, peer_responses, nxt.proposal)
            return nxt.proposal, ready, False

        if peers_opening_task is not None:
            return asyncio.create_task(_run_opening_peers())
        if lead_name and response.agent != lead_name:
            if _settles_question(response):
                return None
            return asyncio.create_task(_run())
        if queue.peek() is not None:
            return asyncio.create_task(_run())
        return None

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

    async def _announce_adjourned(self, conversation_id: str) -> None:
        """Speak a short neutral closer after the council is finished."""
        await self._speak_line(
            conversation_id,
            speaker=CLERK,
            text=_ADJOURN_LINE,
            intent=Intent.SUMMARY,
        )

    async def _speak(self, conversation_id: str, response: Response) -> None:
        await self._speak_line(
            conversation_id,
            speaker=response.agent,
            text=response.text,
            intent=response.intent,
        )

    async def _speak_line(
        self,
        conversation_id: str,
        *,
        speaker: str,
        text: str,
        intent: Intent,
    ) -> None:
        correlation_id = uuid4().hex
        speak = SpeakEvent(
            conversation_id=conversation_id,
            speaker=speaker,
            text=text,
            intent=intent,
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

    def _opening_slots(self, accepted: list[Proposal]) -> list[Proposal]:
        """Lead (if present) plus at most one other member to open the floor.

        When Grok (Brutalist) bid, prefer them as that peer slot over Anthropic/DeepSeek.
        """
        if self._lead is None:
            return accepted[:2]
        lead_name = self._lead.name
        lead = next((p for p in accepted if p.agent == lead_name), None)
        peers = [p for p in accepted if p.agent != lead_name]
        preferred = self._moderator.config.brutalist_name
        peers.sort(key=lambda p: (1 if p.agent == preferred else 0, p.confidence), reverse=True)
        if lead is None:
            return peers[:1]
        return [lead, *peers[:1]]

    async def _offer_lead_reply(
        self,
        conversation: Conversation,
        queue: SpeakingQueue,
        peer_responses: list[Response],
        spoken: Message,
    ) -> bool:
        """Let the Lead respond after a peer. Returns True if Lead was queued."""
        if self._lead is None or self._cancelled:
            return False
        # Don't double-book the Lead if they're already waiting.
        if any(item.agent == self._lead.name for item in queue.snapshot()):
            return True

        conversation_id = conversation.id
        context = self._build_context(
            conversation, queue, peer_responses, Phase.PROPOSAL, speaking_now=spoken
        )
        proposal = await self._lead.propose(context)
        await self._bus.publish(ProposalCreated(conversation_id=conversation_id, proposal=proposal))
        selection = await self._moderator.choose_speakers([proposal], conversation_id)
        if not selection.accepted:
            return False
        accepted = selection.accepted[0]
        queue.enqueue_front(QueuedSpeech(proposal=accepted, is_interrupt=True))
        await self._bus.publish(
            SpeechQueued(
                conversation_id=conversation_id,
                speaker=accepted.agent,
                intent=accepted.intent,
                position=0,
                is_interrupt=True,
            )
        )
        return True

    async def _interrupt_round(
        self,
        conversation: Conversation,
        queue: SpeakingQueue,
        peer_responses: list[Response],
        spoken: Message,
        *,
        min_confidence_override: int | None = None,
    ) -> None:
        """Give every member a chance to raise an interrupt about ``spoken``."""
        conversation_id = conversation.id
        lead_name = self._lead.name if self._lead else None
        # Peers only — the Lead replies via ``_offer_lead_reply``.
        agents = [a for a in self._agents if a.name != lead_name]
        proposals = await self._moderator.request_proposals(
            agents,
            lambda agent: self._build_context(
                conversation, queue, peer_responses, Phase.PROPOSAL, speaking_now=spoken
            ),
            conversation_id,
        )
        selection = await self._moderator.choose_speakers(
            proposals,
            conversation_id,
            interrupt=True,
            min_confidence_override=min_confidence_override,
        )
        for proposal in selection.accepted:
            if self._interrupts_used >= self._max_interrupts:
                break
            self._interrupts_used += 1
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
        """Offer the Lead a wrap-up only when the thread still needs one.

        If a peer already answered (e.g. Grok), the turn ends on them — no
        forced Gemini summary.
        """
        if self._lead is None or len({r.agent for r in peer_responses}) < 2:
            return
        last = peer_responses[-1]
        if last.agent != self._lead.name and _settles_question(last):
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
