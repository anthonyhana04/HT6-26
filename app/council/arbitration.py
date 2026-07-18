"""Pure ranking logic: turn a bag of proposals into an ordered speaker list.

Arbitration contains *no* I/O and *no* LLM calls — it is deterministic, unit-
testable policy. It answers three questions:

1. Who cleared the bar to speak? (should_speak + confidence threshold)
2. Should this collapse to a single speaker? (a confident, dominant answer)
3. In what order should the survivors speak? (lead opens; corrections follow
   their target; otherwise by confidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.models.proposal import Intent, Proposal

# Intents that make sense for the Lead to *open* a discussion with.
_LEAD_OPENING_INTENTS = {Intent.ANSWER, Intent.SUMMARY, Intent.OBSERVATION, Intent.FOLLOW_UP}
# Intents that can legitimately win the floor alone.
_SOLO_INTENTS = {Intent.ANSWER, Intent.SUMMARY}


@dataclass(frozen=True)
class RejectedProposal:
    proposal: Proposal
    reason: str


@dataclass(frozen=True)
class Selection:
    """The outcome of arbitration for one round."""

    accepted: list[Proposal] = field(default_factory=list)
    rejected: list[RejectedProposal] = field(default_factory=list)

    @property
    def is_silent(self) -> bool:
        return not self.accepted


@dataclass(frozen=True)
class ArbitrationPolicy:
    """Tunable thresholds for one arbitration pass."""

    min_confidence: int = 55
    max_speakers: int = 3
    # If the top bid beats the next by this much, a solo answer wins the floor.
    dominance_gap: int = 25


class Arbitrator:
    """Ranks proposals into an ordered, deduplicated speaking list."""

    def __init__(self, lead_name: str) -> None:
        self._lead_name = lead_name

    def select(self, proposals: list[Proposal], policy: ArbitrationPolicy) -> Selection:
        rejected: list[RejectedProposal] = []
        candidates: list[Proposal] = []

        for proposal in proposals:
            if not proposal.should_speak:
                rejected.append(RejectedProposal(proposal, "chose silence"))
            elif proposal.confidence < policy.min_confidence:
                rejected.append(
                    RejectedProposal(proposal, f"confidence {proposal.confidence} < {policy.min_confidence}")
                )
            else:
                candidates.append(proposal)

        if not candidates:
            return Selection(accepted=[], rejected=rejected)

        ranked = sorted(candidates, key=self._rank_key, reverse=True)

        # Solo-answer collapse: a confident, dominant answer speaks alone.
        top = ranked[0]
        if top.intent in _SOLO_INTENTS:
            runner_up = ranked[1] if len(ranked) > 1 else None
            if runner_up is None or (top.confidence - runner_up.confidence) >= policy.dominance_gap:
                for loser in ranked[1:]:
                    rejected.append(RejectedProposal(loser, "deferred to a dominant answer"))
                return Selection(accepted=[top], rejected=rejected)

        # Otherwise admit up to the speaker cap.
        admitted = ranked[: policy.max_speakers]
        for loser in ranked[policy.max_speakers :]:
            rejected.append(RejectedProposal(loser, "speaker cap reached"))

        ordered = self._order(admitted)
        return Selection(accepted=ordered, rejected=rejected)

    # -- ordering ------------------------------------------------------------ #

    def _rank_key(self, proposal: Proposal) -> tuple[int, int, int]:
        """Sort by confidence, then lead priority, then a stable intent weight."""
        lead_priority = 1 if proposal.agent == self._lead_name else 0
        return (proposal.confidence, lead_priority, _intent_weight(proposal.intent))

    def _order(self, proposals: list[Proposal]) -> list[Proposal]:
        """Lead opens; then confidence order; then respect reply targets."""
        lead_openers = [
            p for p in proposals if p.agent == self._lead_name and p.intent in _LEAD_OPENING_INTENTS
        ]
        rest = [p for p in proposals if p not in lead_openers]
        rest.sort(key=lambda p: p.confidence, reverse=True)
        ordered = lead_openers + rest
        return _respect_targets(ordered)


def _intent_weight(intent: Intent) -> int:
    # Tie-breaker only: answers/summaries slightly ahead of reactions.
    order = {
        Intent.ANSWER: 6,
        Intent.SUMMARY: 5,
        Intent.CORRECTION: 4,
        Intent.DISAGREEMENT: 3,
        Intent.QUESTION: 2,
        Intent.OBSERVATION: 1,
        Intent.FOLLOW_UP: 1,
        Intent.AGREEMENT: 0,
    }
    return order.get(intent, 0)


def _respect_targets(ordered: list[Proposal]) -> list[Proposal]:
    """Ensure a proposal replying to peer X is scheduled after X (if present)."""
    names = {p.agent for p in ordered}
    result = list(ordered)
    for _ in range(len(result)):  # bounded passes; converges quickly
        moved = False
        for i, proposal in enumerate(result):
            target = proposal.target
            if target and target in names:
                target_index = next(j for j, p in enumerate(result) if p.agent == target)
                if target_index > i:
                    result.insert(target_index + 1, result.pop(i))
                    moved = True
                    break
        if not moved:
            break
    return result
