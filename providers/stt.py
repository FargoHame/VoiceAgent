"""
Deepgram STT provider.

Implements the SpeechToText protocol.

Design constraints:
  - Opens exactly one WebSocket at startup; keeps it alive for the session.
  - Has no knowledge of the LLM, TTS, or orchestrator.
  - Cancellation is observed via the shared asyncio.Event; it never cancels
    anything itself.
  - Re-connects automatically if the WebSocket drops unexpectedly.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Final

import websockets
from websockets.exceptions import ConnectionClosed

from config import Config

logger = logging.getLogger(__name__)

_URL: Final[str] = (
    "wss://api.deepgram.com/v1/listen"
    "?encoding=linear16"
    "&sample_rate=16000"
    "&channels=1"
    "&endpointing=500"
    "&interim_results=false"
)

# How long to wait before attempting a reconnect after an unexpected drop.
_RECONNECT_DELAY_S: Final[float] = 0.5


class DeepgramSTT:
    """Persistent Deepgram STT connection with automatic reconnection."""

    def __init__(self) -> None:
        self._headers = {"Authorization": f"Token {Config.DEEPGRAM_API_KEY}"}

    # ------------------------------------------------------------------
    # Public interface (SpeechToText protocol)
    # ------------------------------------------------------------------

    async def run(
        self,
        mic: asyncio.Queue[bytes | None],
        out: asyncio.Queue[str],
        cancel: asyncio.Event,
    ) -> None:
        """
        Forward PCM chunks from *mic* to Deepgram; put final transcripts on *out*.
        Reconnects on unexpected drops until *cancel* is set.
        """
        while not cancel.is_set():
            try:
                await self._session(mic, out, cancel)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if cancel.is_set():
                    break
                logger.warning("STT session error (%s); reconnecting in %.1fs", exc, _RECONNECT_DELAY_S)
                await asyncio.sleep(_RECONNECT_DELAY_S)

        logger.info("STT shut down cleanly.")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _session(
        self,
        mic: asyncio.Queue[bytes | None],
        out: asyncio.Queue[str],
        cancel: asyncio.Event,
    ) -> None:
        """Run one WebSocket session until cancellation or connection drop."""
        async with websockets.connect(
            _URL,
            additional_headers=self._headers,
            ping_interval=10,
            ping_timeout=20,
        ) as ws:
            logger.info("STT WebSocket connected.")

            sender_task = asyncio.create_task(self._sender(ws, mic, cancel))
            receiver_task = asyncio.create_task(self._receiver(ws, out, cancel))

            try:
                # Wait for either task to finish; the other is then cancelled.
                done, pending = await asyncio.wait(
                    [sender_task, receiver_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

                # Re-raise any exception from completed tasks.
                for task in done:
                    if not task.cancelled() and task.exception():
                        raise task.exception()  # type: ignore[misc]
            finally:
                sender_task.cancel()
                receiver_task.cancel()

    async def _sender(
        self,
        ws: websockets.WebSocketClientProtocol,
        mic: asyncio.Queue[bytes | None],
        cancel: asyncio.Event,
    ) -> None:
        """Read PCM from the mic queue and forward to Deepgram."""
        cancel_waiter = asyncio.create_task(cancel.wait())
        try:
            while True:
                get_task = asyncio.create_task(mic.get())
                done, _ = await asyncio.wait(
                    [get_task, cancel_waiter],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if cancel_waiter in done:
                    get_task.cancel()
                    await ws.send(json.dumps({"type": "CloseStream"}))
                    return

                chunk = get_task.result()
                if chunk is None:  # Sentinel: microphone shutting down.
                    await ws.send(json.dumps({"type": "CloseStream"}))
                    return

                await ws.send(chunk)
                mic.task_done()
        finally:
            cancel_waiter.cancel()

    async def _receiver(
        self,
        ws: websockets.WebSocketClientProtocol,
        out: asyncio.Queue[str],
        cancel: asyncio.Event,
    ) -> None:
        """Receive Deepgram messages and forward final transcripts to *out*."""
        try:
            async for raw in ws:
                if cancel.is_set():
                    return

                msg = json.loads(raw)
                if not msg.get("is_final", False):
                    continue

                transcript: str = (
                    msg.get("channel", {})
                    .get("alternatives", [{}])[0]
                    .get("transcript", "")
                    .strip()
                )
                if transcript and len(transcript.split()) >= 3:  # Filter out very short false positives.
                    logger.info("STT final: %r", transcript)
                    await out.put(transcript)
        except ConnectionClosed:
            logger.debug("STT WebSocket closed by server.")