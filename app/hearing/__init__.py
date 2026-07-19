"""Always-on hearing: wake phrase + utterance routing into the council."""

from app.hearing.wake import WAKE_PHRASE, contains_wake, strip_wake

__all__ = ["WAKE_PHRASE", "contains_wake", "strip_wake"]
