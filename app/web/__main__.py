"""Run the phone-mic web server:  ``python -m app.web``

Serves an HTTPS page you open on your phone to talk to the council. HTTPS is
required because browsers only allow microphone access in a secure context; a
self-signed certificate is generated automatically on first run (you'll tap
through a one-time "not private" warning on the phone).
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
from pathlib import Path

import uvicorn
from rich.console import Console

from app.config.factory import build_council
from app.config.settings import Settings
from app.main import ConsoleObserver, _print_banner
from app.speech.transcription import ElevenLabsTranscriber
from app.web.server import create_app


def _lan_ip() -> str:
    """Best-effort primary LAN IP (for printing the phone URL)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


def _ensure_cert(cert_dir: Path) -> tuple[Path, Path]:
    """Generate a self-signed cert/key with openssl if they don't exist."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert = cert_dir / "cert.pem"
    key = cert_dir / "key.pem"
    if cert.exists() and key.exists():
        return cert, key
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(key), "-out", str(cert),
            "-days", "365", "-subj", "/CN=aicouncil",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cert, key


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    console = Console()
    settings = Settings()

    if not settings.elevenlabs_api_key:
        console.print("[red]ELEVENLABS_API_KEY is required for speech-to-text.[/]")
        raise SystemExit(1)

    build = build_council(settings, console=console)
    # Keep the console observer alive so the Pi terminal shows the discussion.
    _observer = ConsoleObserver(console, build.bus, verbose=True)  # noqa: F841
    transcriber = ElevenLabsTranscriber(settings.elevenlabs_api_key)

    _print_banner(build)

    host = os.environ.get("AI_COUNCIL_WEB_HOST", "0.0.0.0")
    port = int(os.environ.get("AI_COUNCIL_WEB_PORT", "8443"))
    cert, key = _ensure_cert(Path(os.environ.get("AI_COUNCIL_CERT_DIR", "certs")))

    url = f"https://{_lan_ip()}:{port}"
    console.print(f"[bold green]Open this on your phone:[/] [bold]{url}[/]")
    console.print("[dim](Accept the one-time certificate warning, then tap the mic.)[/]\n")

    app = create_app(build, transcriber)
    uvicorn.run(app, host=host, port=port, ssl_keyfile=str(key), ssl_certfile=str(cert), log_level="warning")


if __name__ == "__main__":
    main()
