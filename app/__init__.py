"""AI Council — a real-time, event-driven multi-agent orchestration engine.

The package is split into strictly layered subpackages:

- ``models``   : immutable data structures passed between layers.
- ``events``   : the event vocabulary and the async event bus.
- ``memory``   : structured conversation memory.
- ``speech``   : the speech abstraction (terminal today, voice/LED tomorrow).
- ``agents``   : council members and their two-phase (propose/generate) brains.
- ``council``  : the pure-Python orchestration core (moderator, queue, scheduler).
- ``config``   : settings, agent profiles and dependency-injection wiring.

Nothing in ``council`` or ``agents`` ever talks to hardware or voice APIs
directly. They only emit events; peripheral layers subscribe.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
