"""Typed application settings, loaded from environment / ``.env``.

All tunables live here so nothing (models, thresholds, voice ids, keys) is
hardcoded in business logic. Missing provider keys are fine — the factory falls
back to the offline mock brain for those members.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- engine behaviour --------------------------------------------------- #
    force_mock: bool = Field(default=False, alias="AI_COUNCIL_FORCE_MOCK")
    min_confidence: int = Field(default=55, alias="AI_COUNCIL_MIN_CONFIDENCE")
    interrupt_confidence: int = Field(default=75, alias="AI_COUNCIL_INTERRUPT_CONFIDENCE")
    max_speakers: int = Field(default=3, alias="AI_COUNCIL_MAX_SPEAKERS")
    max_turns: int = Field(default=8, alias="AI_COUNCIL_MAX_TURNS")
    max_turns_per_agent: int = Field(default=2, alias="AI_COUNCIL_MAX_TURNS_PER_AGENT")

    # --- provider API keys -------------------------------------------------- #
    deepseek_api_key: str | None = Field(default=None, alias="DEEPSEEK_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")

    # --- model selection ---------------------------------------------------- #
    deepseek_model: str = Field(default="deepseek-chat", alias="DEEPSEEK_MODEL")
    anthropic_model: str = Field(default="claude-sonnet-4-6", alias="ANTHROPIC_MODEL")
    gemini_model: str = Field(default="gemini-pro-latest", alias="GEMINI_MODEL")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # --- Speech output ------------------------------------------------------ #
    # "auto"  -> ElevenLabs when a key is present, else terminal
    # "terminal" -> always print (no audio)
    # "elevenlabs" -> force voice output (falls back to terminal if key missing)
    speech_output: str = Field(default="auto", alias="AI_COUNCIL_SPEECH")

    # --- ElevenLabs voices -------------------------------------------------- #
    elevenlabs_api_key: str | None = Field(default=None, alias="ELEVENLABS_API_KEY")
    elevenlabs_model: str = Field(default="eleven_flash_v2_5", alias="ELEVENLABS_MODEL")
    # Output device for spoken audio: a device index or a substring of its name
    # (e.g. "pulse", "pipewire", "USB"). Empty = system/PortAudio default.
    audio_output_device: str | None = Field(default=None, alias="AI_COUNCIL_AUDIO_DEVICE")

    # --- Lighting (WiZ bulb) ------------------------------------------------ #
    # IP address of a WiZ bulb. When set, each speaker lights it in their colour
    # and it pulses with their voice. Empty = no lighting.
    wiz_bulb_ip: str | None = Field(default=None, alias="WIZ_BULB_IP")
    # Turn the bulb off when the session ends.
    wiz_off_on_exit: bool = Field(default=True, alias="WIZ_OFF_ON_EXIT")
    deepseek_voice_id: str | None = Field(default=None, alias="DEEPSEEK_VOICE_ID")
    anthropic_voice_id: str | None = Field(default=None, alias="ANTHROPIC_VOICE_ID")
    gemini_voice_id: str | None = Field(default=None, alias="GEMINI_VOICE_ID")
    groq_voice_id: str | None = Field(default=None, alias="GROQ_VOICE_ID")
