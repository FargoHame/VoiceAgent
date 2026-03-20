"""
Agent orchestrator.

The ONLY module that knows about all pipeline components.

Responsibilities:
  1. Instantiate components.
  2. Create all queues and events.
  3. Inject dependencies (queues / events) into each component.
  4. Run the turn-processing loop.
  5. Coordinate cancellation on barge-in.

It does NOT:
  - Implement any audio, STT, LLM, or TTS logic.
  - Hold references across turns beyond what is needed for cancellation.
  - Leak any queue or event reference outside this module.

Turn lifecycle
--------------
  1. STT fires an `is_final` transcript → text_queue.
  2. brain_loop picks it up.
  3. If audio is in flight, sets barge_in_event → AudioSink drains + clears event.
  4. Cancels any running response_task.
  5. Creates a new response_task: LLM → splitter → tts_queue → TTS → audio_queue.
  6. Goes back to waiting on text_queue.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Final
import queue

from audio_io import AudioManager
from pipeline.splitter import split_sentences
from providers.llm import GroqLLM
from providers.stt import DeepgramSTT
from providers.tts import DeepgramTTS

logger = logging.getLogger(__name__)

# Queue size caps prevent unbounded memory growth.
# Audio chunks are ~2 KB each; 200 ≈ 400 KB buffer — roughly 12 s at 16 kHz.
_AUDIO_QUEUE_MAX: Final[int] = 200
# Sentence queue: at most a handful of sentences ahead.
_SENTENCE_QUEUE_MAX: Final[int] = 16


class AgentOrchestrator:
    """Wire pipeline components together; own the turn lifecycle."""

    def __init__(self) -> None:
        # --- Components (no cross-references between them) ---
        self._audio = AudioManager()
        self._stt = DeepgramSTT()
        self._llm = GroqLLM()
        self._tts = DeepgramTTS()

        # --- Queues (created here, injected into components) ---
        self._mic_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._text_queue: asyncio.Queue[str] = asyncio.Queue()
        self._sentence_queue: asyncio.Queue[str | None] = asyncio.Queue(_SENTENCE_QUEUE_MAX)
        self._audio_queue: asyncio.Queue[bytes | None] = queue.Queue(_AUDIO_QUEUE_MAX)

        # --- Shared cancellation events ---
        # barge_in_event: set when the user speaks mid-playback.
        # Observed by AudioSink to drain and stop; cleared by AudioSink after draining.
        self._barge_in_event: asyncio.Event = asyncio.Event()

        # shutdown_event: set once on shutdown; observed by all long-running tasks.
        self._shutdown_event: asyncio.Event = asyncio.Event()

        # --- Internal state ---
        self._response_task: asyncio.Task | None = None
        self._background_tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start all pipeline components and block until shutdown."""
        # Microphone runs in a plain thread (sounddevice is blocking).
        mic_thread = threading.Thread(
            target=self._audio.start,
            args=(self._mic_queue,),
            daemon=True,
            name="microphone",
        )
        mic_thread.start()

        self._background_tasks = [
            asyncio.create_task(
                self._stt.run(self._mic_queue, self._text_queue, self._shutdown_event),
                name="stt",
            ),
            asyncio.create_task(
                self._tts.run(self._sentence_queue, self._audio_queue, self._shutdown_event),
                name="tts",
            ),
            asyncio.create_task(
                self._audio.run(self._audio_queue, self._barge_in_event),
                name="audio_sink",
            ),
            asyncio.create_task(
                self._brain_loop(),
                name="brain",
            ),
        ]

        logger.info("Agent ready and listening.")
        try:
            await asyncio.gather(*self._background_tasks)
        except asyncio.CancelledError:
            pass

    def shutdown(self) -> None:
        """Signal all components to stop and clean up."""
        logger.info("Shutting down…")
        self._shutdown_event.set()

        if self._response_task and not self._response_task.done():
            self._response_task.cancel()

        for task in self._background_tasks:
            task.cancel()

        # Unblock any components waiting on queues.
        self._mic_queue.put_nowait(None)
        self._sentence_queue.put_nowait(None)
        self._audio_queue.put_nowait(None)

        logger.info("Shutdown complete.")

    # ------------------------------------------------------------------
    # Turn processing
    # ------------------------------------------------------------------

    async def _brain_loop(self) -> None:
        """
        Wait for final transcripts; manage turn lifecycle.

        This is the only coroutine allowed to touch _response_task.
        """
        shutdown_waiter = asyncio.create_task(self._shutdown_event.wait())
        try:
            while True:
                get_task = asyncio.create_task(self._text_queue.get())
                done, _ = await asyncio.wait(
                    [get_task, shutdown_waiter],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if shutdown_waiter in done:
                    get_task.cancel()
                    return

                transcript: str = get_task.result()
                self._text_queue.task_done()

                # Debounce: if another transcript arrives within 300ms, merge them.
                # Deepgram sometimes splits one utterance into two rapid is_final events.
                try:
                    followup = await asyncio.wait_for(self._text_queue.get(), timeout=0.3)
                    transcript = transcript + " " + followup
                    self._text_queue.task_done()
                    logger.info("Merged rapid STT finals: %r", transcript)
                except asyncio.TimeoutError:
                    pass

                logger.info("Turn start: %r", transcript)

                await self._handle_barge_in()
                await self._start_response(transcript)

        except asyncio.CancelledError:
            pass
        finally:
            shutdown_waiter.cancel()

    async def _handle_barge_in(self) -> None:
        """
        If a response is already in-flight, cancel it and signal the audio
        sink to stop playback immediately.
        """
        if self._response_task and not self._response_task.done():
            logger.info("Barge-in detected — cancelling current response.")
            self._response_task.cancel()
            try:
                await self._response_task
            except (asyncio.CancelledError, Exception):
                pass
            self._response_task = None

            # Signal the audio sink to drain its queue.
            self._barge_in_event.set()

            # Drain the sentence queue so TTS doesn't synthesise stale sentences.
            self._drain_queue(self._sentence_queue)

    async def _start_response(self, transcript: str) -> None:
        """Create an async task that runs LLM → splitter → sentence_queue."""
        self._response_task = asyncio.create_task(
            self._produce_sentences(transcript),
            name=f"response:{transcript[:20]}",
        )

    async def _produce_sentences(self, prompt: str) -> None:
        """
        Stream tokens from the LLM, split into sentences, put on sentence_queue.

        Cancellation: asyncio.CancelledError propagates naturally — the LLM
        generator is abandoned and the sentence_queue is left as-is (the
        barge-in handler will drain it).
        """
        try:
            token_stream = self._llm.generate(prompt)
            async for sentence in split_sentences(token_stream):
                if self._shutdown_event.is_set():
                    return
                logger.debug("Sentence ready: %r", sentence[:60])
                await self._sentence_queue.put(sentence)
        except asyncio.CancelledError:
            logger.debug("Response task cancelled mid-stream.")
            raise

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drain_queue(q: asyncio.Queue) -> None:
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except asyncio.QueueEmpty:
                break