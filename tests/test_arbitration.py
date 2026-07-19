"""Unit tests for the deterministic arbitration policy.

Arbitration is pure (no I/O, no LLMs), so it is the natural place to pin down
the engine's most important behaviours with fast, deterministic tests.
"""

from __future__ import annotations

from app.council.arbitration import Arbitrator, ArbitrationPolicy
from app.models.proposal import Intent, Proposal

LEAD = "DeepSeek"


def _p(agent: str, speak: bool, conf: int, intent: Intent, target: str | None = None) -> Proposal:
    return Proposal(agent=agent, should_speak=speak, confidence=conf, intent=intent, reason="t", target=target)


def test_silence_when_nobody_bids() -> None:
    arb = Arbitrator(LEAD)
    result = arb.select(
        [_p("Anthropic", False, 10, Intent.OBSERVATION), _p("Groq", False, 20, Intent.DISAGREEMENT)],
        ArbitrationPolicy(),
    )
    assert result.is_silent
    assert len(result.rejected) == 2


def test_dominant_answer_collapses_to_single_speaker() -> None:
    """A confident, dominant ANSWER wins the floor alone (the '2+2' case)."""
    arb = Arbitrator(LEAD)
    result = arb.select(
        [
            _p(LEAD, True, 95, Intent.ANSWER),
            _p("Gemini", True, 40, Intent.QUESTION),
        ],
        ArbitrationPolicy(min_confidence=55, max_speakers=3, dominance_gap=25),
    )
    assert [p.agent for p in result.accepted] == [LEAD]


def test_open_discussion_admits_several() -> None:
    """Close confidences (an open-ended topic) admit multiple speakers."""
    arb = Arbitrator(LEAD)
    result = arb.select(
        [
            _p(LEAD, True, 82, Intent.ANSWER),
            _p("Anthropic", True, 88, Intent.OBSERVATION),
            _p("Gemini", True, 80, Intent.QUESTION),
            _p("Groq", True, 84, Intent.DISAGREEMENT),
        ],
        ArbitrationPolicy(min_confidence=55, max_speakers=3, dominance_gap=25),
    )
    assert len(result.accepted) == 3  # capped
    assert LEAD in {p.agent for p in result.accepted}


def test_confidence_threshold_filters() -> None:
    arb = Arbitrator(LEAD)
    result = arb.select(
        [_p("Gemini", True, 40, Intent.QUESTION)],
        ArbitrationPolicy(min_confidence=55),
    )
    assert result.is_silent


def test_lead_opens_and_targets_follow_their_referent() -> None:
    """Lead speaks first; a correction targeting Grok is ordered after Grok."""
    arb = Arbitrator(LEAD)
    result = arb.select(
        [
            _p(LEAD, True, 70, Intent.ANSWER),
            _p("Grok", True, 90, Intent.DISAGREEMENT),
            _p("Anthropic", True, 85, Intent.CORRECTION, target="Grok"),
        ],
        ArbitrationPolicy(min_confidence=55, max_speakers=3, dominance_gap=25),
    )
    order = [p.agent for p in result.accepted]
    assert order[0] == LEAD  # lead opens despite lower confidence
    assert order.index("Anthropic") > order.index("Grok")  # correction follows target


def test_preferred_peer_beats_other_peers_in_close_calls() -> None:
    """Grok is preferred over Anthropic when confidences are close."""
    arb = Arbitrator(LEAD, preferred_peer="Grok")
    result = arb.select(
        [
            _p(LEAD, True, 70, Intent.ANSWER),
            _p("Anthropic", True, 82, Intent.OBSERVATION),
            _p("Grok", True, 78, Intent.DISAGREEMENT),
        ],
        ArbitrationPolicy(min_confidence=55, max_speakers=3, dominance_gap=25),
    )
    order = [p.agent for p in result.accepted]
    assert order[0] == LEAD
    assert order.index("Grok") < order.index("Anthropic")


def test_non_lead_solo_does_not_silence_preferred_peer() -> None:
    """Anthropic's dominant answer cannot mute Grok; Lead solo still can."""
    arb = Arbitrator(LEAD, preferred_peer="Grok")
    peer_solo = arb.select(
        [
            _p("Anthropic", True, 95, Intent.ANSWER),
            _p("Grok", True, 60, Intent.DISAGREEMENT),
        ],
        ArbitrationPolicy(min_confidence=55, max_speakers=3, dominance_gap=25),
    )
    assert {p.agent for p in peer_solo.accepted} == {"Anthropic", "Grok"}

    lead_solo = arb.select(
        [
            _p(LEAD, True, 95, Intent.ANSWER),
            _p("Grok", True, 60, Intent.DISAGREEMENT),
        ],
        ArbitrationPolicy(min_confidence=55, max_speakers=3, dominance_gap=25),
    )
    assert [p.agent for p in lead_solo.accepted] == [LEAD]
