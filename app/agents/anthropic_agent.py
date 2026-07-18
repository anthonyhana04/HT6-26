"""Anthropic-backed council member (the Thinker)."""

from __future__ import annotations

from typing import Any

from app.agents.base import AgentProfile, BaseAgent


class AnthropicAgent(BaseAgent):
    """Council member powered by the Anthropic Messages API.

    Anthropic has no dedicated JSON mode, so in ``json_mode`` we nudge the model
    with a system suffix and rely on :meth:`BaseAgent._parse_proposal` to be
    forgiving of any surrounding text.
    """

    def __init__(self, profile: AgentProfile, api_key: str, *, client: Any | None = None) -> None:
        super().__init__(profile)
        if client is None:
            from anthropic import AsyncAnthropic  # lazy: keeps the SDK optional

            client = AsyncAnthropic(api_key=api_key)
        self._client = client

    async def _complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if json_mode:
            system = f"{system}\n\nReturn ONLY the raw JSON object with no surrounding text."

        response = await self._client.messages.create(
            model=self._profile.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
