"""DeepSeek-backed council member (the Lead).

DeepSeek exposes an OpenAI-compatible Chat Completions API, so we reuse the
``openai`` SDK pointed at DeepSeek's base URL. Only the endpoint and default
model differ from a vanilla OpenAI agent.
"""

from __future__ import annotations

from typing import Any

from app.agents.base import AgentProfile, BaseAgent

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekAgent(BaseAgent):
    """Council member powered by the DeepSeek API (OpenAI-compatible schema)."""

    def __init__(
        self,
        profile: AgentProfile,
        api_key: str,
        *,
        base_url: str = DEEPSEEK_BASE_URL,
        client: Any | None = None,
    ) -> None:
        super().__init__(profile)
        if client is None:
            from openai import AsyncOpenAI  # lazy: DeepSeek speaks the OpenAI schema

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
