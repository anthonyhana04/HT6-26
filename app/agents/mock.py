"""A deterministic, offline "brain" so the engine runs without any API keys.

The :class:`MockAgent` is not a scripted personality — it makes the *same kind*
of autonomous decision a real agent does: given the context, decide whether to
speak, with what intent and confidence. Its heuristics are tuned to demonstrate
the core promise of the engine:

* trivial/factual prompts → only the Lead answers, everyone else stays silent;
* open-ended/decision prompts → several members engage, in character.

It overrides :meth:`propose`/:meth:`generate` directly (rather than
``_complete``) because it reasons over the structured :class:`AgentContext`
instead of a text prompt.
"""

from __future__ import annotations

import re

from app.agents.base import AgentProfile, BaseAgent
from app.agents.complexity import classify as _classify
from app.models.context import AgentContext
from app.models.proposal import Intent, Proposal
from app.models.response import Response

# The mock decides behaviour by ROLE (stable) rather than provider name, so it
# keeps working when providers are swapped between roles.
ROLE_LEAD = "Lead"
ROLE_THINKER = "Thinker"
ROLE_SECOND_GUESSER = "Second Guesser"
ROLE_BRUTALIST = "Brutalist"

# For the few cross-references (addressing a specific peer), the mock needs peer
# *names*. These reflect the default roster and are demo-only — kept local so
# agents never depend on the config layer. If they no longer match, reactions
# simply don't fire (the engine degrades gracefully).
_LEAD_NAME = "Gemini"
_BRUTALIST_NAME = "Grok"


class MockAgent(BaseAgent):
    """A rule-based council member for offline runs, demos and tests."""

    def __init__(self, profile: AgentProfile) -> None:
        super().__init__(profile)

    async def _complete(self, system: str, user: str, *, json_mode: bool, max_tokens: int, temperature: float) -> str:
        # Not used: propose/generate are overridden. Present to satisfy the ABC.
        return ""

    # -- phase 1 ------------------------------------------------------------- #

    async def propose(self, context: AgentContext) -> Proposal:
        if context.is_closing:
            return self._closing_proposal(context)
        if context.speaking_now is not None:
            return self._react_proposal(context)
        return self._opening_proposal(context)

    def _opening_proposal(self, context: AgentContext) -> Proposal:
        complexity = _classify(context.user_message.content)
        role = self.role

        if complexity == "trivial":
            if self.is_lead:
                return self._bid(95, Intent.ANSWER, "Direct factual question — I'll take it.")
            return self._pass("Trivial question; the Lead should answer alone.")

        # Open-ended / decision topic: members engage in character, by role.
        if self.is_lead:
            return self._bid(82, Intent.ANSWER, "Frame the question and give an initial take.")
        if role == ROLE_THINKER:
            return self._bid(88, Intent.OBSERVATION, "There are long-term implications worth surfacing.")
        if role == ROLE_SECOND_GUESSER:
            return self._bid(80, Intent.QUESTION, "Some assumptions here deserve challenging.", target=_LEAD_NAME)
        if role == ROLE_BRUTALIST:
            return self._bid(84, Intent.DISAGREEMENT, "This needs a blunt reality check.", target=_LEAD_NAME)
        return self._pass("Nothing to add.")

    def _closing_proposal(self, context: AgentContext) -> Proposal:
        """The Lead may take the last word to summarize a multi-voice thread."""
        distinct_speakers = {r.agent for r in context.peer_responses}
        if self.is_lead and len(distinct_speakers) >= 2:
            return self._bid(80, Intent.SUMMARY, "Tie the discussion together for the user.")
        return self._pass("No closing remark needed.")

    def _react_proposal(self, context: AgentContext) -> Proposal:
        """Decide whether to raise an interrupt about what's being spoken."""
        speaking = context.speaking_now
        assert speaking is not None

        # Don't react to yourself, on trivial topics, or when you're already
        # scheduled to speak (raising an interrupt would double-book you).
        already_queued = {q.agent for q in context.queue}
        if (
            speaking.speaker == self.name
            or self.name in already_queued
            or _classify(context.user_message.content) == "trivial"
        ):
            return self._pass("Nothing to interrupt about.")

        # Sparse reactions — most interrupt rounds stay quiet.
        role = self.role
        if role == ROLE_BRUTALIST and speaking.speaker == _LEAD_NAME:
            return self._bid(
                72,
                Intent.DISAGREEMENT,
                "Lead's take is soft — one pushback.",
                target=_LEAD_NAME,
            )
        if role == ROLE_SECOND_GUESSER and speaking.speaker == _LEAD_NAME:
            return self._bid(78, Intent.QUESTION, "The Lead's framing skips an edge case.", target=_LEAD_NAME)
        return self._pass("Content to let it stand.")

    # -- phase 2 ------------------------------------------------------------- #

    async def generate(self, context: AgentContext) -> Response:
        proposal = context.committed_proposal
        intent = proposal.intent if proposal else Intent.OBSERVATION
        target = proposal.target if proposal else None
        topic = _topic(context.user_message.content)
        text = self._voice(intent, topic, context)
        return Response(agent=self.name, text=text, intent=intent, target=target)

    def _voice(self, intent: Intent, topic: str, context: AgentContext) -> str:
        """Produce an in-character line. Deterministic, persona-flavoured."""
        complexity = _classify(context.user_message.content)

        if complexity == "trivial" and self.is_lead:
            return _answer_trivial(context.user_message.content)

        if self.is_lead:
            if intent is Intent.SUMMARY:
                return f"So to pull this together on {topic}: weigh the upside against the concrete risks, and take one small reversible step first."
            return f"Let's frame this. On {topic}, the core question is whether the upside justifies what you'd be giving up. Here's my initial read."
        if self.role == ROLE_THINKER:
            return f"Stepping back, the interesting part of {topic} is the second-order effects — what this commits you to a year from now, not just next week."
        if self.role == ROLE_SECOND_GUESSER:
            ref = (context.committed_proposal.target if context.committed_proposal and context.committed_proposal.target else "we")
            return f"I'd question an assumption {ref} is making about {topic}: what happens in the case where it doesn't go to plan? Have we actually priced that in?"
        if self.role == ROLE_BRUTALIST:
            target = context.committed_proposal.target if context.committed_proposal else None
            if intent is Intent.CORRECTION:
                who = target or "look"
                return f"Hold on — {who}, you skipped the hard part of {topic}: can it actually be executed? A plan you can't run is just a wish."
            return f"Honestly? Most of the hand-wringing about {topic} is noise. Decide what actually moves the needle, cut the rest, and ship. Talk is cheap."
        return f"A quick observation on {topic}."

    # -- helpers ------------------------------------------------------------- #

    def _bid(self, confidence: int, intent: Intent, reason: str, *, target: str | None = None) -> Proposal:
        return Proposal(agent=self.name, should_speak=True, confidence=confidence, intent=intent, reason=reason, target=target)

    def _pass(self, reason: str) -> Proposal:
        return Proposal(agent=self.name, should_speak=False, confidence=15, intent=Intent.OBSERVATION, reason=reason)


# --------------------------------------------------------------------------- #
# Free functions
# --------------------------------------------------------------------------- #


def _topic(text: str) -> str:
    cleaned = text.strip().rstrip("?.!").lower()
    for lead in ("i'm considering ", "i am considering ", "should i ", "should we ", "what do you think about ", "thinking about "):
        if cleaned.startswith(lead):
            cleaned = cleaned[len(lead) :]
            break
    return (cleaned[:60] + "…") if len(cleaned) > 60 else (cleaned or "this")


def _answer_trivial(text: str) -> str:
    match = re.search(r"(\d+)\s*([\+\-\*/x])\s*(\d+)", text)
    if match:
        a, op, b = int(match.group(1)), match.group(2), int(match.group(3))
        result = {"+" : a + b, "-": a - b, "*": a * b, "x": a * b, "/": (a / b if b else float("nan"))}[op]
        pretty = int(result) if float(result).is_integer() else result
        return f"That's {pretty}."
    return "Good question — the short answer is: it depends on a single key detail, which I'm happy to pin down."
