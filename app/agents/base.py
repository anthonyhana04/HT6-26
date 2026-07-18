"""The council-member abstraction and the two-phase brain shared by all agents.

Every provider (OpenAI, Anthropic, Gemini, Groq) is only responsible for one
thing: turning a (system, user) prompt into text via :meth:`BaseAgent._complete`.
Everything else — building the proposal and generation prompts, parsing the
lightweight proposal JSON, shaping the final :class:`Response` — lives here so
behaviour is identical and testable across providers.

Phase 1 (``propose``) is intentionally cheap: a small, JSON-only completion that
returns *whether* and *why* the agent wants to speak. Phase 2 (``generate``) is
only ever run for accepted speakers.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from app.agents.complexity import token_budget
from app.models.context import AgentContext, Phase
from app.models.proposal import Intent, Proposal
from app.models.response import Response

logger = logging.getLogger("ai_council.agents")

_INTENT_VALUES = ", ".join(i.value for i in Intent)


class AgentProfile(BaseModel):
    """Static configuration describing one council member.

    Personas and model ids come from configuration, never hardcoded in logic.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    role: str
    persona: str
    is_lead: bool = False
    model: str = ""
    voice_id: str | None = None

    # These are outer sanity ceilings. The *effective* budget used per call is
    # tightened further, per message, by app.agents.complexity.token_budget —
    # a trivial question never spends anywhere near these caps.
    proposal_max_tokens: int = Field(default=320, ge=1)
    proposal_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    generation_max_tokens: int = Field(default=420, ge=1)
    generation_temperature: float = Field(default=0.75, ge=0.0, le=2.0)


class BaseAgent(ABC):
    """A council member with a proposal phase and a generation phase."""

    def __init__(self, profile: AgentProfile) -> None:
        self._profile = profile

    # -- identity ------------------------------------------------------------ #

    @property
    def profile(self) -> AgentProfile:
        return self._profile

    @property
    def name(self) -> str:
        return self._profile.name

    @property
    def role(self) -> str:
        return self._profile.role

    @property
    def is_lead(self) -> bool:
        return self._profile.is_lead

    # -- provider hook ------------------------------------------------------- #

    @abstractmethod
    async def _complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Return a completion for the given prompt.

        Providers implement only this. When ``json_mode`` is true the provider
        should request structured/JSON output where the SDK supports it.
        """

    # -- phase 1: proposal --------------------------------------------------- #

    async def propose(self, context: AgentContext) -> Proposal:
        """Cheaply decide whether (and how) to contribute."""
        system = self._proposal_system_prompt()
        user = self._proposal_user_prompt(context)
        max_tokens = token_budget(
            context.user_message.content, phase="proposal", ceiling=self._profile.proposal_max_tokens
        )
        try:
            raw = await self._complete(
                system,
                user,
                json_mode=True,
                max_tokens=max_tokens,
                temperature=self._profile.proposal_temperature,
            )
            return self._parse_proposal(raw)
        except Exception as exc:  # noqa: BLE001 — a failing agent must not stall the round
            # Expected in practice (quota, balance, retired model). Log concise;
            # full traceback only when DEBUG is enabled.
            logger.warning("%s could not propose (%s); defaulting to silence", self.name, _brief(exc))
            logger.debug("Proposal failure detail for %s", self.name, exc_info=exc)
            return Proposal(
                agent=self.name,
                should_speak=False,
                confidence=0,
                intent=Intent.OBSERVATION,
                reason="Proposal generation failed.",
            )

    # -- phase 2: generation ------------------------------------------------- #

    async def generate(self, context: AgentContext) -> Response:
        """Produce the full spoken contribution (accepted speakers only)."""
        system = self._generation_system_prompt()
        user = self._generation_user_prompt(context)
        intent = context.committed_proposal.intent if context.committed_proposal else Intent.OBSERVATION
        target = context.committed_proposal.target if context.committed_proposal else None
        # A reactive turn (peers already spoke, or this is a targeted reply)
        # needs room to engage by name even if the original question read as
        # simple — only the true opening turn on a trivial question stays tight.
        is_reactive = bool(context.peer_responses) or target is not None
        max_tokens = token_budget(
            context.user_message.content,
            phase="generation",
            ceiling=self._profile.generation_max_tokens,
            force_open=is_reactive,
        )
        try:
            text = await self._complete(
                system,
                user,
                json_mode=False,
                max_tokens=max_tokens,
                temperature=self._profile.generation_temperature,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s could not generate (%s)", self.name, _brief(exc))
            logger.debug("Generation failure detail for %s", self.name, exc_info=exc)
            text = "(…lost my train of thought.)"
        return Response(agent=self.name, text=text.strip(), intent=intent, target=target)

    # -- prompt construction ------------------------------------------------- #

    def _proposal_system_prompt(self) -> str:
        return (
            f"{_COUNCIL_OVERVIEW}\n\n"
            f"You are {self.name}, the {self.role} of the council.\n"
            f"{self._profile.persona}\n\n"
            "PROPOSAL PHASE.\n"
            "You are NOT answering yet. You are only deciding whether you should "
            "speak at all, and if so, why. Silence is a valid and often correct "
            "choice. Do not speak merely to agree, echo, or pad the discussion.\n\n"
            "Guidelines:\n"
            "- If the message is simple or factual (e.g. a quick calculation or "
            "definition), usually only ONE member should answer. If someone else "
            "is clearly better positioned, stay silent.\n"
            "- Only bid to speak if you add something genuinely new: a correction, "
            "a missing consideration, a challenge, a needed answer, or a summary.\n"
            "- Prefer high confidence only when your contribution is clearly valuable.\n"
            f"- Valid intents: {_INTENT_VALUES}.\n"
            "- 'target' is the name of another member you are responding to, or null.\n\n"
            "Respond with ONLY a JSON object, no prose, of exactly this shape:\n"
            '{"should_speak": bool, "confidence": 0-100, '
            '"intent": one of the intents, "reason": short string, '
            '"target": member name or null}\n'
            "Keep 'reason' to ONE concise sentence (under 20 words). Output the "
            "JSON and nothing else."
        )

    def _proposal_user_prompt(self, context: AgentContext) -> str:
        return self._render_context(context, phase=Phase.PROPOSAL)

    def _generation_system_prompt(self) -> str:
        return (
            f"{_COUNCIL_OVERVIEW}\n\n"
            f"You are {self.name}, the {self.role} of the council.\n"
            f"{self._profile.persona}\n\n"
            "GENERATION PHASE.\n"
            "The moderator selected you to speak. Deliver your contribution as if "
            "talking aloud in a live board meeting:\n"
            "- Match your length to the question. A simple/factual question gets "
            "ONE short sentence — do not pad it with caveats or context. A "
            "genuinely complex or open-ended topic can earn 2-4 sentences, never "
            "more.\n"
            "- Address peers by name when correcting, challenging or building on them.\n"
            "- Do NOT restate the question or repeat what others already said.\n"
            "- Stay in character. No meta-commentary, no markdown, no lists.\n"
            "- Speak only your part; other members will handle theirs."
        )

    def _generation_user_prompt(self, context: AgentContext) -> str:
        base = self._render_context(context, phase=Phase.GENERATION)
        if context.committed_proposal is not None:
            base += (
                f"\n\nYour committed intent: {context.committed_proposal.intent.value}."
                f"\nWhy you wanted to speak: {context.committed_proposal.reason}"
            )
            if context.committed_proposal.target:
                base += f"\nYou are responding to: {context.committed_proposal.target}."
        return base

    def _render_context(self, context: AgentContext, *, phase: Phase) -> str:
        parts: list[str] = []

        if context.history:
            transcript = "\n".join(m.as_transcript_line() for m in context.history[-12:])
            parts.append(f"Earlier conversation:\n{transcript}")

        parts.append(f"Current user message:\nUser: {context.user_message.content}")

        if context.discussion:
            live = "\n".join(m.as_transcript_line() for m in context.discussion)
            parts.append(f"Discussion so far this turn:\n{live}")

        if context.peer_responses:
            peers = "\n".join(f"{r.agent}: {r.text}" for r in context.peer_responses)
            parts.append(f"What peers just said this turn:\n{peers}")

        if context.speaking_now is not None:
            parts.append(
                "Currently being spoken (you may raise an interrupt to respond "
                f"AFTER it finishes):\n{context.speaking_now.as_transcript_line()}"
            )

        if context.queue:
            queued = ", ".join(
                f"{q.agent}({q.intent}{'*' if q.is_interrupt else ''})" for q in context.queue
            )
            parts.append(f"Already queued to speak: {queued}")

        if context.is_closing:
            parts.append(
                "The discussion is winding down. If a brief closing SUMMARY or a "
                "genuinely useful final point would help, bid to speak; otherwise "
                "stay silent."
            )

        return "\n\n".join(parts)

    # -- parsing ------------------------------------------------------------- #

    def _parse_proposal(self, raw: str) -> Proposal:
        """Robustly parse possibly-messy model JSON into a :class:`Proposal`."""
        data = _extract_json_object(raw)
        if data is None:
            # The most common cause is truncation mid-string (a long "reason"
            # ran past max_tokens), which breaks strict JSON parsing even though
            # should_speak/confidence/intent were fully written. Recover what we
            # can instead of discarding a perfectly good bid.
            data = _recover_partial_fields(raw)
        if data is None:
            logger.warning("%s returned unparseable proposal: %r", self.name, raw[:160])
            return Proposal(
                agent=self.name,
                should_speak=False,
                confidence=0,
                intent=Intent.OBSERVATION,
                reason="Unparseable proposal.",
            )

        intent = _coerce_intent(data.get("intent"))
        target = data.get("target")
        if isinstance(target, str) and target.strip().lower() in {"", "null", "none", self.name.lower()}:
            target = None

        return Proposal(
            agent=self.name,
            should_speak=bool(data.get("should_speak", False)),
            confidence=data.get("confidence", 0),
            intent=intent,
            reason=str(data.get("reason", "")).strip() or "(no reason given)",
            target=target if isinstance(target, str) else None,
        )


# --------------------------------------------------------------------------- #
# Shared prompt fragments & parsing helpers
# --------------------------------------------------------------------------- #

_COUNCIL_OVERVIEW = (
    "You are one member of the AI Council — a small group of distinct AI "
    "personalities who discuss the user's ideas together, like colleagues around "
    "a table. Members are: Gemini (Lead — keeps things moving, answers direct "
    "questions, summarizes), Anthropic (Thinker — deep reasoning, systems "
    "thinking, long-term implications, subtle flaws), DeepSeek (Second Guesser — "
    "questions assumptions, finds edge cases, offers alternatives) and Groq "
    "(Brutalist — blunt, execution-first, calls out weak ideas). The group does "
    "NOT all talk at once; members speak only when they truly add value."
)


def _brief(exc: Exception) -> str:
    """A short, single-line description of an exception for user-facing logs."""
    message = " ".join(str(exc).split())
    if len(message) > 200:
        message = message[:200] + "…"
    return message or type(exc).__name__


def _coerce_intent(value: object) -> Intent:
    if isinstance(value, Intent):
        return value
    if isinstance(value, str):
        try:
            return Intent(value.strip().upper())
        except ValueError:
            return Intent.OBSERVATION
    return Intent.OBSERVATION


def _extract_json_object(raw: str) -> dict | None:
    """Extract the first JSON object from a string that may contain extra text."""
    if not raw:
        return None
    text = raw.strip()
    # Strip common code-fence wrappers.
    if text.startswith("```"):
        text = text.strip("`")
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1 :]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost brace pair.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


_FIELD_PATTERNS = {
    "should_speak": re.compile(r'"should_speak"\s*:\s*(true|false)', re.IGNORECASE),
    "confidence": re.compile(r'"confidence"\s*:\s*(\d+(?:\.\d+)?)'),
    "intent": re.compile(r'"intent"\s*:\s*"([A-Za-z_]+)"'),
    # Reason/target are quoted strings which may be truncated (no closing quote).
    # Capture up to the closing quote if present, else to end of string.
    "reason": re.compile(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)(?:"|$)'),
    "target": re.compile(r'"target"\s*:\s*(null|"(?:[^"\\]|\\.)*")'),
}


def _recover_partial_fields(raw: str) -> dict | None:
    """Best-effort field extraction from JSON truncated mid-generation.

    Only succeeds if the three fields the moderator actually depends on
    (``should_speak``, ``confidence``, ``intent``) are present; ``reason`` and
    ``target`` are recovered opportunistically. Returns ``None`` if even the
    critical fields cannot be found (nothing salvageable).
    """
    text = raw.strip()
    if not text or "should_speak" not in text:
        return None

    result: dict[str, object] = {}
    for field, pattern in _FIELD_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            continue
        value = match.group(1)
        if field == "should_speak":
            result[field] = value.lower() == "true"
        elif field == "confidence":
            result[field] = float(value)
        elif field == "target" and value != "null":
            result[field] = value.strip('"')
        elif field != "target":
            result[field] = value

    if "should_speak" not in result or "confidence" not in result or "intent" not in result:
        return None
    result.setdefault("reason", "(truncated)")
    return result
