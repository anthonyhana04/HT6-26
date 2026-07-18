"""The moderator: pure-Python chairperson of the council.

The moderator is explicitly **not** an LLM. It owns every decision about *who*
speaks and *in what order*, keeping the discussion fair and bounded:

* requests proposals from all members (in parallel);
* ranks them via the :class:`~app.council.arbitration.Arbitrator`;
* enforces fairness (a per-turn speaking cap per member);
* decides which interrupts are worth honouring;
* emits ``ProposalCreated`` / ``ProposalAccepted`` / ``ProposalRejected`` so the
  reasoning is fully observable on the event bus.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

from app.agents.base import BaseAgent
from app.council.arbitration import Arbitrator, ArbitrationPolicy, Selection
from app.events.event_bus import EventBus
from app.events.event_types import ProposalAccepted, ProposalCreated, ProposalRejected
from app.models.context import AgentContext
from app.models.proposal import Proposal

# Builds the per-agent context for a proposal round (scheduler supplies this).
# Pure and synchronous: assembling context never does I/O.
ContextBuilder = Callable[[BaseAgent], AgentContext]


@dataclass(frozen=True)
class ModeratorConfig:
    min_confidence: int = 55
    interrupt_confidence: int = 75
    max_speakers: int = 3
    max_turns_per_agent: int = 2
    dominance_gap: int = 25


class Moderator:
    """Runs proposal rounds and selects speakers. Holds no conversation state
    beyond per-turn fairness counters, which are reset each user message."""

    def __init__(self, bus: EventBus, lead_name: str, config: ModeratorConfig) -> None:
        self._bus = bus
        self._config = config
        self._lead = lead_name
        self._arbitrator = Arbitrator(lead_name)
        self._turn_counts: Counter[str] = Counter()

    # -- fairness bookkeeping ------------------------------------------------ #

    def reset_turn(self) -> None:
        """Clear per-user-message speaking counters."""
        self._turn_counts.clear()

    def _record_spoken(self, agent: str) -> None:
        self._turn_counts[agent] += 1

    def _at_cap(self, agent: str) -> bool:
        # The Lead chairs the discussion and is exempt from the per-member
        # fairness cap — it may always answer, follow up, or close. The global
        # ``max_turns`` ceiling still bounds the conversation as a whole.
        if agent == self._lead:
            return False
        return self._turn_counts[agent] >= self._config.max_turns_per_agent

    # -- proposal round ------------------------------------------------------ #

    async def request_proposals(
        self,
        agents: Iterable[BaseAgent],
        build_context: ContextBuilder,
        conversation_id: str,
    ) -> list[Proposal]:
        """Ask every agent for a proposal concurrently and announce each."""
        agents = list(agents)

        async def _one(agent: BaseAgent) -> Proposal:
            context = build_context(agent)
            return await agent.propose(context)

        proposals = await asyncio.gather(*(_one(agent) for agent in agents))
        for proposal in proposals:
            await self._bus.publish(ProposalCreated(conversation_id=conversation_id, proposal=proposal))
        return list(proposals)

    # -- selection ----------------------------------------------------------- #

    async def choose_speakers(
        self,
        proposals: list[Proposal],
        conversation_id: str,
        *,
        interrupt: bool = False,
    ) -> Selection:
        """Apply fairness + arbitration, emit decisions, return the selection."""
        policy = self._policy(interrupt=interrupt)

        # Fairness pre-filter: drop members who already hit their turn cap.
        fair: list[Proposal] = []
        pre_rejected: list[tuple[Proposal, str]] = []
        for proposal in proposals:
            if proposal.should_speak and self._at_cap(proposal.agent):
                pre_rejected.append((proposal, "reached fair-speaking cap this turn"))
            else:
                fair.append(proposal)

        selection = self._arbitrator.select(fair, policy)

        # Emit rejections (fairness + arbitration) and acceptances, in order.
        for proposal, reason in pre_rejected:
            await self._bus.publish(
                ProposalRejected(conversation_id=conversation_id, proposal=proposal, reason=reason)
            )
        for rejected in selection.rejected:
            await self._bus.publish(
                ProposalRejected(
                    conversation_id=conversation_id, proposal=rejected.proposal, reason=rejected.reason
                )
            )
        for position, proposal in enumerate(selection.accepted):
            self._record_spoken(proposal.agent)
            await self._bus.publish(
                ProposalAccepted(conversation_id=conversation_id, proposal=proposal, position=position)
            )

        return selection

    def _policy(self, *, interrupt: bool) -> ArbitrationPolicy:
        if interrupt:
            return ArbitrationPolicy(
                min_confidence=self._config.interrupt_confidence,
                max_speakers=1,  # one interrupt at a time keeps order sane
                dominance_gap=self._config.dominance_gap,
            )
        return ArbitrationPolicy(
            min_confidence=self._config.min_confidence,
            max_speakers=self._config.max_speakers,
            dominance_gap=self._config.dominance_gap,
        )
