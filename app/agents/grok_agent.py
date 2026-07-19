"""xAI Grok-backed council member (the Brutalist).

xAI exposes an OpenAI-compatible Chat Completions API at ``api.x.ai``, so we
reuse the ``openai`` SDK pointed at that base URL — same pattern as DeepSeek.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import AgentProfile, BaseAgent

XAI_BASE_URL = "https://api.x.ai/v1"


class GrokAgent(BaseAgent):
    """Council member powered by xAI Grok (OpenAI-compatible schema)."""

    def __init__(
        self,
        profile: AgentProfile,
        api_key: str,
        *,
        base_url: str = XAI_BASE_URL,
        client: Any | None = None,
    ) -> None:
        super().__init__(profile)
        if client is None:
            from openai import AsyncOpenAI  # lazy: xAI speaks the OpenAI schema

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
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
        kwargs: dict[str, Any] = {}
        # Some Grok models ignore/soft-fail json_object; keep it when asked.
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._client.chat.completions.create(
            model=self._profile.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return response.choices[0].message.content or ""
