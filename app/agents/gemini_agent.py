"""Google Gemini-backed council member (the Second Guesser)."""

from __future__ import annotations

from typing import Any

from app.agents.base import AgentProfile, BaseAgent


class GeminiAgent(BaseAgent):
    """Council member powered by the Google Gen AI SDK (``google-genai``)."""

    def __init__(self, profile: AgentProfile, api_key: str, *, client: Any | None = None) -> None:
        super().__init__(profile)
        if client is None:
            from google import genai  # lazy: keeps the SDK optional

            client = genai.Client(api_key=api_key)
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
        from google.genai import types

        def build_config(*, disable_thinking: bool, tokens: int) -> "types.GenerateContentConfig":
            kwargs: dict = dict(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=tokens,
                response_mime_type="application/json" if json_mode else "text/plain",
            )
            if disable_thinking:
                # Current Gemini "flash" models default to a thinking phase that
                # can silently consume the whole token budget before any visible
                # output — leaving truncated/empty responses. These council calls
                # are short and structured, so we turn thinking off where allowed.
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            return types.GenerateContentConfig(**kwargs)

        try:
            response = await self._client.aio.models.generate_content(
                model=self._profile.model,
                contents=user,
                config=build_config(disable_thinking=True, tokens=max_tokens),
            )
        except Exception:  # noqa: BLE001 — "pro" models mandate thinking; can't disable
            # Give internal thinking generous headroom so it doesn't truncate the
            # visible JSON/answer, then let the parser extract what we need.
            response = await self._client.aio.models.generate_content(
                model=self._profile.model,
                contents=user,
                config=build_config(disable_thinking=False, tokens=max(max_tokens * 8, 4096)),
            )
        return response.text or ""
