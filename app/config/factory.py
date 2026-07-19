"""Composition root: wire settings + profiles into a ready :class:`Council`.

This is the *only* place that decides which concrete implementations get used
(real provider vs mock brain, terminal vs ElevenLabs speech). Every other
module depends on abstractions and receives its collaborators by injection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rich.console import Console

from app.agents.anthropic_agent import AnthropicAgent
from app.agents.base import AgentProfile, BaseAgent
from app.agents.deepseek_agent import DeepSeekAgent
from app.agents.gemini_agent import GeminiAgent
from app.agents.groq_agent import GroqAgent
from app.agents.mock import MockAgent
from app.config.profiles import ANTHROPIC, DEEPSEEK, GEMINI, GROQ, LEAD_NAME, default_profiles
from app.config.settings import Settings
from app.council.council import Council
from app.council.moderator import Moderator, ModeratorConfig
from app.council.scheduler import Scheduler
from app.events.event_bus import EventBus
from app.lighting.service import LightService
from app.lighting.wiz import WizLightBackend
from app.memory.history import History
from app.speech.base import SpeechBackend
from app.speech.elevenlabs import ElevenLabsSpeech
from app.speech.player import TerminalPlayer
from app.speech.service import SpeechService
from app.speech.style import rgb_for

logger = logging.getLogger("ai_council.factory")


@dataclass(frozen=True)
class CouncilBuild:
    """The assembled engine plus metadata useful to the shell/CLI."""

    council: Council
    bus: EventBus
    console: Console
    agent_modes: dict[str, str]  # name -> "live (<model>)" | "mock"
    speech_mode: str  # "elevenlabs (<model>)" | "terminal"
    lighting_mode: str  # "wiz · <ip>" | "off"


def build_council(settings: Settings | None = None, *, console: Console | None = None) -> CouncilBuild:
    settings = settings or Settings()
    console = console or Console()
    bus = EventBus()

    profiles = default_profiles(settings)
    agents: list[BaseAgent] = []
    agent_modes: dict[str, str] = {}

    for name, profile in profiles.items():
        agent, mode = _build_agent(name, profile, settings)
        agents.append(agent)
        agent_modes[name] = mode

    backend, speech_mode = _build_speech_backend(settings, profiles, console)
    speech_service = SpeechService(bus, backend)

    light_service, lighting_mode = _build_light_service(settings, profiles, bus)

    moderator = Moderator(
        bus,
        lead_name=LEAD_NAME,
        config=ModeratorConfig(
            min_confidence=settings.min_confidence,
            interrupt_confidence=settings.interrupt_confidence,
            max_speakers=settings.max_speakers,
            max_turns_per_agent=settings.max_turns_per_agent,
        ),
    )
    scheduler = Scheduler(bus, moderator, agents, max_turns=settings.max_turns)
    history = History()

    council = Council(
        bus=bus,
        agents=agents,
        moderator=moderator,
        scheduler=scheduler,
        history=history,
        speech_service=speech_service,
        light_service=light_service,
    )
    return CouncilBuild(
        council=council,
        bus=bus,
        console=console,
        agent_modes=agent_modes,
        speech_mode=speech_mode,
        lighting_mode=lighting_mode,
    )


def _build_light_service(
    settings: Settings,
    profiles: dict[str, AgentProfile],
    bus: EventBus,
) -> tuple[LightService | None, str]:
    """Attach a WiZ bulb to the speech event stream when one is configured."""
    ip = (settings.wiz_bulb_ip or "").strip()
    if not ip:
        return None, "off"
    try:
        backend = WizLightBackend(ip, off_on_close=settings.wiz_off_on_exit)
        colors = {name: rgb_for(name) for name in profiles}
        service = LightService(bus, backend, colors)
        return service, f"wiz · {ip}"
    except Exception as exc:  # noqa: BLE001 — missing dep / bad IP → no lighting
        logger.warning("WiZ lighting init failed (%s); running without lights", exc)
        return None, "off"


def _build_speech_backend(
    settings: Settings,
    profiles: dict[str, AgentProfile],
    console: Console,
) -> tuple[SpeechBackend, str]:
    """Choose the speech backend from configuration.

    ``AI_COUNCIL_SPEECH`` selects the mode: ``auto`` uses ElevenLabs when a key
    is present (otherwise terminal), ``elevenlabs`` forces voice (falling back
    to terminal if the key is missing), and ``terminal`` is always text-only.
    """
    mode = (settings.speech_output or "auto").strip().lower()
    want_voice = mode == "elevenlabs" or (mode == "auto" and bool(settings.elevenlabs_api_key))

    if want_voice and settings.elevenlabs_api_key:
        # name -> configured voice_id (None entries get auto-assigned a unique voice).
        voice_ids: dict[str, str | None] = {name: p.voice_id for name, p in profiles.items()}
        try:
            backend = ElevenLabsSpeech(
                settings.elevenlabs_api_key,
                voice_ids,
                model_id=settings.elevenlabs_model,
                device=settings.audio_output_device or None,
                console=console,
            )
            return backend, f"elevenlabs · {settings.elevenlabs_model}"
        except Exception as exc:  # noqa: BLE001 — missing SDK → fall back to text
            logger.warning("ElevenLabs init failed (%s); using terminal speech", exc)
    elif want_voice:
        logger.warning("ELEVENLABS_API_KEY not set; using terminal speech")

    return TerminalPlayer(console=console), "terminal"


def _build_agent(name: str, profile: AgentProfile, settings: Settings) -> tuple[BaseAgent, str]:
    """Pick a live provider agent when a key is present, else the mock brain."""
    if settings.force_mock:
        return MockAgent(profile), "mock (forced)"

    key, agent_cls = {
        DEEPSEEK: (settings.deepseek_api_key, DeepSeekAgent),
        ANTHROPIC: (settings.anthropic_api_key, AnthropicAgent),
        GEMINI: (settings.google_api_key, GeminiAgent),
        GROQ: (settings.groq_api_key, GroqAgent),
    }[name]

    if not key:
        return MockAgent(profile), "mock (no key)"

    try:
        agent = agent_cls(profile, key)  # type: ignore[call-arg]
        return agent, f"live · {profile.model}"
    except Exception as exc:  # noqa: BLE001 — missing SDK or bad init → degrade
        logger.warning("Falling back to mock for %s: %s", name, exc)
        return MockAgent(profile), "mock (init failed)"
