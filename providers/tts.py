"""
Deepgram TTS provider.

Implements the TextToSpeech protocol.

Design constraints:
  - Opens exactly one WebSocket at startup; reuses it across all turns.
  - Accepts sentences from a queue; never pulls from the LLM directly.
  - Cancellation via asyncio.Event: drains the sentence queue and discards
    any audio already in-flight, then returns.
  - No knowledge of the LLM, STT, or orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Final
import queue
import websockets
from websockets.exceptions import ConnectionClosed
import time
from config import Config

logger = logging.getLogger(__name__)

_URL: Final[str] = (
    "wss://api.deepgram.com/v1/speak"
    "?model=aura-asteria-en"
    "&encoding=linear16"
    "&sample_rate=16000"
)

_RECONNECT_DELAY_S: Final[float] = 0.5


class DeepgramTTS:
    """Persistent Deepgram TTS connection, sentence-level streaming."""

    def __init__(self) -> None:
        self._headers = {"Authorization": f"Token {Config.DEEPGRAM_API_KEY}"}

    # ------------------------------------------------------------------
    # Public interface (TextToSpeech protocol)
    # ------------------------------------------------------------------

    async def run(
        self,
        sentences: asyncio.Queue[str | None],
        out: queue.Queue[bytes | None],
        cancel: asyncio.Event,
    ) -> None:
        """
        Read sentences from *sentences*, synthesise audio, put PCM on *out*.
        Reconnects on unexpected drops until *cancel* is set.
        """
        while not cancel.is_set():
            try:
                await self._session(sentences, out, cancel)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if cancel.is_set():
                    break
                logger.warning("TTS session error (%s); reconnecting in %.1fs", exc, _RECONNECT_DELAY_S)
                await asyncio.sleep(_RECONNECT_DELAY_S)

        logger.info("TTS shut down cleanly.")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _session(
        self,
        sentences: asyncio.Queue[str | None],
        out: asyncio.Queue[bytes | None],
        cancel: asyncio.Event,
    ) -> None:
        """Run one WebSocket session, processing sentences until cancelled."""
        async with websockets.connect(
            _URL,
            additional_headers=self._headers,
            ping_interval=10,
            ping_timeout=20,
        ) as ws:
            logger.info("TTS WebSocket connected.")

            cancel_waiter = asyncio.create_task(cancel.wait())
            try:
                while True:
                    # Wait for the next sentence or cancellation.
                    get_task = asyncio.create_task(sentences.get())
                    done, _ = await asyncio.wait(
                        [get_task, cancel_waiter],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if cancel_waiter in done:
                        get_task.cancel()
                        self._drain(sentences)
                        return

                    sentence: str | None = get_task.result()
                    sentences.task_done()

                    if sentence is None:  # Sentinel: pipeline shutting down.
                        return

                    await self._synthesise_one(ws, sentence, out, cancel)

                    # After cancellation during synthesis, flush and return.
                    if cancel.is_set():
                        self._drain(sentences)
                        return
            finally:
                cancel_waiter.cancel()

    async def _synthesise_one(
        self,
        ws: websockets.WebSocketClientProtocol,
        sentence: str,
        out: queue.Queue[bytes | None],
        cancel: asyncio.Event,
    ) -> None:
        """Send one sentence to Deepgram and collect audio until Flushed."""
        t0 = time.monotonic()
        first_chunk = True
        logger.info("TTS send: %r", sentence[:60])

        await ws.send(json.dumps({"type": "Speak", "text": sentence}))
        await ws.send(json.dumps({"type": "Flush"}))

        async for raw in ws:
            if cancel.is_set():
                return

            if isinstance(raw, bytes):
                if first_chunk:
                    logger.info("TTS first audio chunk: %.3fs after send", time.monotonic() - t0)
                    first_chunk = False
                try:    
                    out.put_nowait(raw)
                except queue.Full:
                    logger.warning("Audio queue full; dropping chunk.")
            else:
                msg = json.loads(raw)
                if msg.get("type") == "Flushed":
                    logger.info("TTS flushed: %.3fs total", time.monotonic() - t0)
                    return

    @staticmethod
    def _drain(q: asyncio.Queue) -> None:
        """Empty a queue without blocking (best-effort, non-blocking)."""
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except asyncio.QueueEmpty:
                break