"""
Audio I/O manager.

Implements both MicrophoneSource and AudioSink protocols.

Design constraints:
  - Pure hardware I/O — no knowledge of STT, TTS, LLM, or orchestrator.
  - Microphone runs in a dedicated OS thread (sounddevice callback model).
  - Playback is an async coroutine driven by a queue.
  - Cancellation of playback via asyncio.Event: drains the queue, stops output.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import sounddevice as sd

from config import Config

logger = logging.getLogger(__name__)


class AudioManager:
    """Hardware microphone capture and speaker playback."""

    # ------------------------------------------------------------------
    # MicrophoneSource protocol
    # ------------------------------------------------------------------

    def start(self, out: asyncio.Queue[bytes | None]) -> None:
        """
        Blocking.  Run in a dedicated thread.

        Opens a sounddevice RawInputStream, forwards PCM chunks to *out*.
        Returns only when the stream becomes inactive (hardware error / shutdown).
        """
        def _callback(
            indata: bytes,
            frames: int,
            time: object,
            status: sd.CallbackFlags,
        ) -> None:
            if status:
                logger.warning("Mic status: %s", status)
            # put_nowait is safe from a C-extension callback thread.
            out.put_nowait(bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=Config.AUDIO.RATE,
                blocksize=Config.AUDIO.CHUNK,
                channels=Config.AUDIO.CHANNELS,
                dtype="int16",
                callback=_callback,
            ):
                logger.info("Microphone active.")
                # Block until the stream becomes inactive.
                while True:
                    sd.sleep(200)
        except Exception as exc:
            logger.error("Microphone error: %s", exc)
        finally:
            # Signal downstream that the mic is gone.
            out.put_nowait(None)

    # ------------------------------------------------------------------
    # AudioSink protocol
    # ------------------------------------------------------------------

    async def run(
        self,
        audio: queue.Queue[bytes | None],
        cancel: asyncio.Event,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._playback_thread, audio, cancel, loop)

    def _playback_thread(
        self,
        audio: queue.Queue[bytes | None],
        cancel: asyncio.Event,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Runs entirely in a thread — no event loop bouncing per chunk."""
        stream = sd.RawOutputStream(
            samplerate=Config.AUDIO.RATE,
            blocksize=Config.AUDIO.CHUNK,
            channels=Config.AUDIO.CHANNELS,
            dtype="int16",
        )
        stream.start()
        logger.info("Speaker active.")

        try:
            while True:
                if cancel.is_set():
                    self._drain(audio)
                    cancel.clear()
                    logger.debug("Playback interrupted; queue drained.")

                try:
                    chunk = audio.get_nowait()
                except queue.Empty:
                    # Nothing ready — yield briefly to avoid busy-spin.
                    import time
                    time.sleep(0.005)
                    continue

                if chunk is None:
                    break

                stream.write(chunk)
                

        except Exception as exc:
            logger.error("Playback error: %s", exc)
        finally:
            stream.stop()
            stream.close()
            logger.info("Speaker closed.")
        # ------------------------------------------------------------------
        # Private helpers
        # ------------------------------------------------------------------

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except asyncio.QueueEmpty:
                break