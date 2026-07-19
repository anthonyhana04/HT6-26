"""Physical lighting layer — a pure subscriber to the speech event stream.

Nothing here is imported by the council core. The composition root wires a
:class:`LightService` onto the bus when a bulb is configured; everything else
is untouched.
"""

from app.lighting.base import RGB, LightBackend
from app.lighting.service import LightService
from app.lighting.wiz import WizLightBackend

__all__ = ["RGB", "LightBackend", "LightService", "WizLightBackend"]
