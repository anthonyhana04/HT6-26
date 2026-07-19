"""AI Council — CLI entry point.

Run with:  ``python -m app.main``

The CLI is a *subscriber*, not a driver. It feeds user input to the
:class:`~app.council.council.Council` and renders the engine's event stream as a
live meeting log. Because it only listens to events, the same engine will one
day drive voices and LEDs with zero changes here.
"""

from __future__ import annotations

import asyncio
import logging

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config.factory import CouncilBuild, build_council
from app.config.settings import Settings
from app.events.event_bus import EventBus
from app.events.event_types import (
    ConversationEnded,
    InterruptRequest,
    ProposalAccepted,
    ProposalCreated,
    ProposalRejected,
    SpeechQueued,
    UserMessage,
)
from app.speech.style import color_for

_HELP = (
    "[bold]Commands[/]\n"
    "  /help      show this help\n"
    "  /verbose   toggle the moderator's reasoning log\n"
    "  /members   list council members and their backends\n"
    "  /quit      leave the council\n\n"
    "Try a trivial prompt ([italic]\"what is 2+2?\"[/]) — only the Lead should "
    "answer. Then try something open-ended ([italic]\"I'm considering quitting "
    "school to start a company\"[/]) — watch several members engage."
)


class ConsoleObserver:
    """Renders orchestration events as a dim, readable meeting log.

    It subscribes only to the event bus — exactly the seam a future LED or voice
    peripheral would use.
    """

    def __init__(self, console: Console, bus: EventBus, *, verbose: bool = True) -> None:
        self._console = console
        self.verbose = verbose
        bus.subscribe(ProposalCreated, self._on_proposal)
        bus.subscribe(ProposalAccepted, self._on_accepted)
        bus.subscribe(ProposalRejected, self._on_rejected)
        bus.subscribe(SpeechQueued, self._on_queued)
        bus.subscribe(InterruptRequest, self._on_interrupt)

    def _tag(self, name: str) -> str:
        return f"[{color_for(name)}]{name}[/]"

    def _on_proposal(self, event: ProposalCreated) -> None:
        if not self.verbose:
            return
        p = event.proposal
        if p.should_speak:
            target = f" → {p.target}" if p.target else ""
            self._console.print(
                f"  [dim]·[/] {self._tag(p.agent)} [dim]bids[/] "
                f"[green]{p.confidence}[/] [dim]{p.intent.value}{target} — {p.reason}[/]"
            )
        else:
            self._console.print(f"  [dim]·[/] {self._tag(p.agent)} [dim]passes — {p.reason}[/]")

    def _on_accepted(self, event: ProposalAccepted) -> None:
        if not self.verbose:
            return
        p = event.proposal
        self._console.print(
            f"  [green]✓[/] {self._tag(p.agent)} [dim]scheduled #{event.position} "
            f"({p.intent.value})[/]"
        )

    def _on_rejected(self, event: ProposalRejected) -> None:
        if not self.verbose:
            return
        self._console.print(
            f"  [red dim]✗ {event.proposal.agent} — {event.reason}[/]"
        )

    def _on_queued(self, event: SpeechQueued) -> None:
        if event.is_interrupt:
            self._console.print(
                f"  [yellow]⤵ interrupt:[/] {self._tag(event.speaker)} "
                f"[dim]queued next ({event.intent.value})[/]"
            )

    def _on_interrupt(self, event: InterruptRequest) -> None:
        if not self.verbose:
            return
        self._console.print(
            f"  [yellow dim]! {event.agent} requests interrupt — {event.reason}[/]"
        )


def _print_banner(build: CouncilBuild) -> None:
    console = build.console
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Member", style="bold")
    table.add_column("Role")
    table.add_column("Backend", style="dim")
    roles = {a.name: a.role for a in build.council.members}
    for name, mode in build.agent_modes.items():
        table.add_row(f"[{color_for(name)}]{name}[/]", roles.get(name, ""), mode)

    console.print(
        Panel(
            table,
            title="[bold]AI Council[/]",
            subtitle="[dim]a real-time multi-agent orchestration engine[/]",
            border_style="blue",
        )
    )
    console.print(f"[dim]Speech: {build.speech_mode}   Lights: {build.lighting_mode}[/]")
    console.print("[dim]Type /help for commands. Ctrl-C or /quit to exit.[/]\n")


async def _read_line(console: Console, prompt: str) -> str | None:
    """Read one line without blocking the event loop; ``None`` on EOF."""
    try:
        return await asyncio.to_thread(console.input, prompt)
    except (EOFError, KeyboardInterrupt):
        return None


async def run() -> None:
    logging.basicConfig(level=logging.WARNING)
    settings = Settings()
    console = Console()
    build = build_council(settings, console=console)
    observer = ConsoleObserver(console, build.bus, verbose=True)

    _print_banner(build)
    council = build.council

    try:
        while True:
            text = await _read_line(console, "[bold green]You[/] › ")
            if text is None:
                break
            text = text.strip()
            if not text:
                continue

            if text.startswith("/"):
                if _handle_command(text, console, build, observer):
                    continue
                break  # /quit

            console.print()
            await council.ask(text)
            console.print()
    finally:
        await council.shutdown()
        console.print("\n[dim]Council adjourned.[/]")


def _handle_command(text: str, console: Console, build: CouncilBuild, observer: ConsoleObserver) -> bool:
    """Handle a ``/command``. Returns False only for ``/quit``."""
    command = text.split()[0].lower()
    if command in {"/quit", "/exit", "/q"}:
        return False
    if command == "/help":
        console.print(Panel(_HELP, border_style="dim", title="Help"))
    elif command == "/verbose":
        observer.verbose = not observer.verbose
        console.print(f"[dim]Moderator log {'on' if observer.verbose else 'off'}.[/]")
    elif command == "/members":
        for agent in build.council.members:
            console.print(
                f"  [{color_for(agent.name)}]{agent.name}[/] — {agent.role} "
                f"[dim]({build.agent_modes.get(agent.name, '?')})[/]"
            )
    else:
        console.print(f"[red]Unknown command:[/] {command}  [dim](try /help)[/]")
    return True


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
