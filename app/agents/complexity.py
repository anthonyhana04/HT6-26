"""A cheap, shared heuristic for how much a question deserves.

The whole point of the two-phase architecture is to avoid spending tokens where
they aren't needed. This module answers one narrow question — "roughly how
much effort does this message warrant?" — and both the offline
:class:`~app.agents.mock.MockAgent` and every live provider agent use it to size
their token budgets, so a "what is 2+2?" costs a sentence and a "should I quit
school to start a company?" still stays to a tight spoken turn — never a rant.

This is deliberately *not* a moderator concern: the moderator decides *who*
speaks; this decides *how much room* whoever speaks should be given.
"""

from __future__ import annotations

import re
from typing import Literal

Complexity = Literal["trivial", "open"]

_ARITHMETIC_RE = re.compile(r"\d+\s*[\+\-\*/x]\s*\d+")
_SIMPLE_STARTS = (
    "what is", "what's", "who is", "who's", "when ", "where ", "define",
    "how many", "capital of",
)
_DECISION_MARKERS = (
    "should i", "should we", "thinking about", "considering", "quit", "quitting",
    "start a company", "startup", "worth it", "what do you think", "opinion",
    "idea", "advice", "decide", "invest", "risk", "future", "strategy", "plan",
    "pros and cons", "trade-off", "tradeoff", "career", "life",
)

# Token ceilings per complexity tier. These are the *effective* caps used by
# BaseAgent; an AgentProfile's own max_tokens fields act as an outer sanity
# ceiling on top of these (see BaseAgent.propose/generate).
PROPOSAL_TOKENS: dict[Complexity, int] = {"trivial": 150, "open": 280}
# Spoken turns must stay short. ~80 tokens ≈ 1–2 sentences; ~180 ≈ a tight 2–3.
GENERATION_TOKENS: dict[Complexity, int] = {"trivial": 80, "open": 180}


def classify(text: str) -> Complexity:
    """Classify a user message as ``"trivial"`` or ``"open"``.

    Trivial: quick factual/arithmetic asks that deserve a short, single-voice
    answer. Open: decisions, opinions, or anything substantial enough that a
    real discussion (and therefore more generation budget) is warranted.
    """
    t = text.lower().strip()
    if _ARITHMETIC_RE.search(t):
        return "trivial"
    if any(marker in t for marker in _DECISION_MARKERS):
        return "open"
    words = t.split()
    if len(words) <= 7 and (t.endswith("?") or t.startswith(_SIMPLE_STARTS)):
        return "trivial"
    return "open" if len(words) > 12 else "trivial"


def token_budget(
    text: str, *, phase: Literal["proposal", "generation"], ceiling: int, force_open: bool = False
) -> int:
    """The token budget for ``text`` at ``phase``, capped by ``ceiling``.

    ``ceiling`` is the agent profile's own configured max — this function only
    ever *tightens* it for simple messages, never loosens it beyond what the
    profile allows.

    ``force_open`` overrides a "trivial" classification to "open". Use this for
    reactive turns (an agent responding to peers who already spoke, e.g. a
    CORRECTION or interrupt) — those inherently need room to engage by name
    even when the original question's wording looked simple.
    """
    complexity: Complexity = "open" if force_open else classify(text)
    table = PROPOSAL_TOKENS if phase == "proposal" else GENERATION_TOKENS
    return min(ceiling, table[complexity])
