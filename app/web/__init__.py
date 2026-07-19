"""Web microphone front-end — the phone-as-mic input into the council.

A pure *input* peripheral: it transcribes recorded audio and calls the same
``council.ask`` the CLI uses. Voices and lights (the output side) are untouched.
"""

from app.web.server import create_app

__all__ = ["create_app"]
