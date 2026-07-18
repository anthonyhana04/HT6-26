"""The speech abstraction: emit ``SpeakEvent``, let a backend render it."""

from app.speech.base import SpeechBackend, SpeechChunk
from app.speech.elevenlabs import ElevenLabsSpeech
from app.speech.player import TerminalPlayer
from app.speech.service import SpeechService

__all__ = [
    "SpeechBackend",
    "SpeechChunk",
    "TerminalPlayer",
    "ElevenLabsSpeech",
    "SpeechService",
]
