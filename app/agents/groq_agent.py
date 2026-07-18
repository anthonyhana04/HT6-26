"""Groq (Llama)-backed council member (the Brutalist)."""

from __future__ import annotations

from typing import Any

from app.agents.base import AgentProfile, BaseAgent


class GroqAgent(BaseAgent):
    """Council member powered by the Groq API (OpenAI-compatible schema)."""

    def __init__(self, profile: AgentProfile, api_key: str, *, client: Any | None = None) -> None:
        super().__init__(profile)
        if client is None:
            from groq import AsyncGroq  # lazy: keeps the SDK optional

            client = AsyncGroq(api_key=api_key)
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
