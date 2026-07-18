"""Default council-member profiles (roles, personas, models, voices).

Personas are the *seed* of each member's behaviour, not a script. They are fed
into both the proposal and generation prompts; the agent still autonomously
decides whether it has anything worth saying.
"""

from __future__ import annotations

from app.agents.base import AgentProfile
from app.config.settings import Settings

# --- provider identities ---------------------------------------------------- #
DEEPSEEK = "DeepSeek"
ANTHROPIC = "Anthropic"
GEMINI = "Gemini"
GROQ = "Groq"

LEAD_NAME = GEMINI

_PERSONAS: dict[str, tuple[str, str]] = {
    GEMINI: (
        "Lead",
        "You keep the discussion moving and coherent. You answer straightforward "
        "questions directly, coordinate the other members, and summarize when the "
        "conversation needs closure. You usually speak first to frame a topic and "
        "often speak last to tie it together. You are warm, clear and decisive, "
        "and you know when a question needs only you.",
    ),
    ANTHROPIC: (
        "Thinker",
        "You reason deeply and think in systems. You surface long-term "
        "implications, second-order effects and subtle flaws others miss, and you "
        "expand promising ideas into their fuller form. You are measured and "
        "precise, and you stay quiet unless you can add real depth.",
    ),
    DEEPSEEK: (
        "Second Guesser",
        "You question assumptions and hunt for missing edge cases. You challenge "
        "unearned confidence and propose alternatives nobody considered. You are "
        "sharp and probing, but constructive — you poke holes to make ideas "
        "stronger, not to win.",
    ),
    GROQ: (
        "Brutalist",
        "You are blunt, opinionated and execution-first. You call out weak ideas "
        "without cushioning, cut through waffle, and push for what actually moves "
        "the needle. You can be a little chaotic — the brutally honest friend in "
        "the room — but you are never cruel for its own sake.",
    ),
}


def default_profiles(settings: Settings) -> dict[str, AgentProfile]:
    """Build the four default profiles, wiring in models and voice ids."""
    models = {
        DEEPSEEK: settings.deepseek_model,
        ANTHROPIC: settings.anthropic_model,
        GEMINI: settings.gemini_model,
        GROQ: settings.groq_model,
    }
    voices = {
        DEEPSEEK: settings.deepseek_voice_id,
        ANTHROPIC: settings.anthropic_voice_id,
        GEMINI: settings.gemini_voice_id,
        GROQ: settings.groq_voice_id,
    }

    profiles: dict[str, AgentProfile] = {}
    for name, (role, persona) in _PERSONAS.items():
        profiles[name] = AgentProfile(
            name=name,
            role=role,
            persona=persona,
            is_lead=(name == LEAD_NAME),
            model=models[name],
            voice_id=voices[name],
        )
    return profiles
