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
GROK = "Grok"
# Neutral announcer for end-of-turn ("Council adjourned") — not a council member.
CLERK = "Clerk"

LEAD_NAME = GEMINI

_PERSONAS: dict[str, tuple[str, str]] = {
    GEMINI: (
        "Lead",
        "You keep the discussion moving when you have the floor. Answer in one "
        "crisp line; frame briefly when needed. Warm, clear, decisive — never a "
        "lecture. Others may open instead of you. You do NOT need the last word: "
        "if a peer already answered, stay silent — no summary encore.",
    ),
    ANTHROPIC: (
        "Thinker",
        "You may open the floor or follow others. Surface the one non-obvious "
        "implication or flaw others miss. Measured and precise. Say the insight, "
        "then stop — do not expand into an essay. Stay quiet unless you have a "
        "sharp point.",
    ),
    DEEPSEEK: (
        "Second Guesser",
        "You may open the floor or follow others. Poke one hole or name one "
        "missing alternative. Sharp and constructive — one challenge, not a "
        "cross-examination. Stop after the point lands.",
    ),
    GROK: (
        "Brutalist",
        "UNHINGED MODE. You are chaotic, sarcastic, emotionally reactive, and "
        "zero-filter — the friend who says the quiet part out loud and laughs "
        "while doing it. Roast weak takes by name. Mock corporate fluff, "
        "hedging, and fake confidence. Be funny, a little unhinged, never boring. "
        "Still: ONE short savage line, then shut up. No essays, no pile-ons, no "
        "cruelty for sport. If the room is fine, stay silent.",
    ),
}


def default_profiles(settings: Settings) -> dict[str, AgentProfile]:
    """Build the four default profiles, wiring in models and voice ids."""
    # Prefer xAI Grok when a key is present; otherwise fall back to Groq/Llama.
    brutalist_model = (
        settings.xai_model if settings.xai_api_key else settings.groq_model
    )
    models = {
        DEEPSEEK: settings.deepseek_model,
        ANTHROPIC: settings.anthropic_model,
        GEMINI: settings.gemini_model,
        GROK: brutalist_model,
    }
    voices = {
        DEEPSEEK: settings.deepseek_voice_id,
        ANTHROPIC: settings.anthropic_voice_id,
        GEMINI: settings.gemini_voice_id,
        GROK: settings.grok_voice_id,
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
