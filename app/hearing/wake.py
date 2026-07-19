"""Wake-phrase helpers for always-on listening.

Accepted wakes (case-insensitive):
  * ``hey council`` / ``hi council`` / ``ok council`` / …
  * bare ``council``

After a wake, the lamp goes white until a real question arrives (or timeout).
"""

from __future__ import annotations

import re

WAKE_PHRASE = "hey council"

# Optional greeting + "council", or bare "council" as its own word.
_WAKE_ANYWHERE = re.compile(
    r"(?:(?:hey|hi|ok|okay|yo|hello|ey)\s+)?council\b",
    re.IGNORECASE,
)
# Leading wake (with optional trailing punctuation) for stripping.
_WAKE_PREFIX = re.compile(
    r"^\s*(?:(?:hey|hi|ok|okay|yo|hello|ey)[\s,]+)?council[\s,!.?]*",
    re.IGNORECASE,
)


def contains_wake(text: str) -> bool:
    return bool(_WAKE_ANYWHERE.search(text or ""))


def strip_wake(text: str) -> str:
    """Remove a leading wake phrase; return the remaining command (may be empty)."""
    if not text:
        return ""
    cleaned = _WAKE_PREFIX.sub("", text, count=1).strip()
    if cleaned == text.strip() and contains_wake(text):
        # Wake mid-clip: keep only what follows the wake token.
        parts = _WAKE_ANYWHERE.split(text, maxsplit=1)
        cleaned = parts[-1] if len(parts) > 1 else ""
        cleaned = re.sub(r"^[\s,!.?]+", "", cleaned).strip()
    return cleaned
