"""Non-interactive demonstration of the orchestration engine.

Runs a scripted set of prompts through the real :class:`Council` (in mock mode
by default, so it needs no API keys) and prints the full meeting log. Handy for
demos, screenshots and as a smoke test.

    AI_COUNCIL_FORCE_MOCK=true python scripts/demo.py
"""

from __future__ import annotations

import asyncio
import os

from rich.console import Console
from rich.rule import Rule

from app.config.factory import build_council
from app.config.settings import Settings
from app.main import ConsoleObserver

PROMPTS = [
    "What is 2+2?",
    "I'm considering quitting school to start a company.",
]


async def main() -> None:
    # Default to mock so the demo runs offline; respect an explicit override.
    os.environ.setdefault("AI_COUNCIL_FORCE_MOCK", "true")

    console = Console()
    build = build_council(Settings(), console=console)
    ConsoleObserver(console, build.bus, verbose=True)

    for prompt in PROMPTS:
        console.print(Rule(f"[bold green]You[/] › {prompt}"))
        await build.council.ask(prompt)
        console.print()

    await build.council.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
