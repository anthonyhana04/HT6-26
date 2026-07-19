"""FastAPI app: tap-to-talk phone mic → council.

Flow:

    tap to speak → tap to send
      → POST /api/hearing/utterance
      → Transcriber → council.ask(text)

If the council is already talking, the first tap fires barge-in (stop speech +
white lamp); the send submits the new question as a fresh turn.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from app.config.factory import CouncilBuild
from app.events.event_types import ListeningStateChanged
from app.speech.transcription import Transcriber

logger = logging.getLogger("ai_council.web")

_STATIC = Path(__file__).parent / "static"


def create_app(build: CouncilBuild, transcriber: Transcriber) -> FastAPI:
    council = build.council
    bus = build.bus
    turn_lock = asyncio.Lock()
    listening_state = {"value": "idle"}  # idle | awaiting_command | busy

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await council.shutdown()
        await transcriber.aclose()

    app = FastAPI(title="AI Council Mic", lifespan=lifespan)

    async def set_state(state: str) -> None:
        if listening_state["value"] == state:
            return
        listening_state["value"] = state
        await bus.publish(ListeningStateChanged(state=state))

    async def _wait_for_turn_slot(*, timeout: float = 4.0) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while turn_lock.locked():
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(0.05)
        return True

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(
            _STATIC / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "speech": build.speech_mode,
                "lights": build.lighting_mode,
                "stt": transcriber.name,
                "busy": turn_lock.locked() or listening_state["value"] == "busy",
                "listening": listening_state["value"],
            }
        )

    @app.post("/api/hearing/barge-in")
    async def barge_in() -> JSONResponse:
        """Cut off current speech when the user taps to interrupt."""
        if listening_state["value"] != "busy" and not turn_lock.locked():
            return JSONResponse({"status": "noop", "listening": listening_state["value"]})
        await council.barge_in(reason="user tapped to interrupt")
        await set_state("awaiting_command")
        return JSONResponse({"status": "interrupted", "listening": "awaiting_command"})

    @app.post("/api/hearing/utterance")
    async def utterance(audio: UploadFile = File(...)) -> JSONResponse:
        data = await audio.read()
        if not data:
            return JSONResponse({"error": "empty audio"}, status_code=400)

        filename = audio.filename or "audio.wav"
        logger.info(
            "Utterance upload: %s (%d bytes, content-type=%s)",
            filename,
            len(data),
            audio.content_type or "?",
        )
        # Phone Safari sometimes posts a near-empty MediaRecorder blob; never
        # bother ElevenLabs with that (it returns a noisy "corrupted" 400).
        if len(data) < 1000:
            return JSONResponse(
                {
                    "error": "Recording too short or stale mic page — hard-refresh, then hold longer.",
                    "status": "too_short",
                    "bytes": len(data),
                },
                status_code=400,
            )

        # Interrupt an in-flight turn, then take the new question.
        if turn_lock.locked() or listening_state["value"] == "busy":
            await council.barge_in(reason="user interrupted")
            if not await _wait_for_turn_slot():
                return JSONResponse(
                    {"status": "busy", "error": "Could not interrupt in time."},
                    status_code=409,
                )

        try:
            text = await transcriber.transcribe(data, filename=filename)
        except Exception:  # noqa: BLE001
            logger.exception("Transcription failed (%s, %d bytes)", filename, len(data))
            return JSONResponse({"error": "Could not transcribe audio."}, status_code=502)

        text = (text or "").strip()
        if not text:
            await set_state("idle")
            return JSONResponse({"transcript": "", "status": "no_speech"})

        await set_state("busy")
        asyncio.create_task(_run_turn(council, turn_lock, set_state, text))
        return JSONResponse({"transcript": text, "status": "submitted", "listening": "busy"})

    @app.post("/api/voice")
    async def voice_alias(audio: UploadFile = File(...)) -> JSONResponse:
        return await utterance(audio)

    return app


async def _run_turn(council, lock: asyncio.Lock, set_state, text: str) -> None:
    async with lock:
        try:
            await council.ask(text)
        except Exception:  # noqa: BLE001
            logger.exception("Council turn failed for %r", text)
        finally:
            await set_state("idle")
