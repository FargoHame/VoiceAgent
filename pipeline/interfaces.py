"""
Pipeline component interfaces.

Every stage in the voice pipeline implements exactly one of these protocols.
No component imports another component — they communicate only through queues
and observe cancellation only through the shared asyncio.Event they are given.

Dependency graph (enforced by these protocols):
  AudioManager  -->  [mic_queue]  -->  STT
  STT           -->  [text_queue] -->  Orchestrator
  Orchestrator  -->  LLM stream   -->  SentenceSplitter
  SentenceSplitter --> [tts_queue] --> TTS
  TTS           -->  [audio_queue] --> AudioManager (playback)
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Protocol, runtime_checkable


@runtime_checkable
class MicrophoneSource(Protocol):
    """Captures raw PCM from hardware and puts bytes onto a queue."""

    def start(self, out: asyncio.Queue[bytes | None]) -> None:
        """
        Blocking; run in a dedicated thread via asyncio.to_thread or threading.Thread.
        Puts raw PCM chunks onto *out*. Sends None as a sentinel on shutdown.
        """
        ...


@runtime_checkable
class SpeechToText(Protocol):
    """Consumes raw PCM, produces final transcripts."""

    async def run(
        self,
        mic: asyncio.Queue[bytes | None],
        out: asyncio.Queue[str],
        cancel: asyncio.Event,
    ) -> None:
        """
        Reads from *mic*, writes final transcript strings to *out*.
        Must return cleanly when *cancel* is set.
        Responsible for maintaining its own warm WebSocket connection.
        """
        ...


@runtime_checkable
class LanguageModel(Protocol):
    """Streams text tokens given a prompt."""

    def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Returns an async generator that yields token strings.
        Caller is responsible for cancellation (just stop iterating).
        """
        ...


@runtime_checkable
class TextToSpeech(Protocol):
    """Consumes sentences, produces PCM audio chunks."""

    async def run(
        self,
        sentences: asyncio.Queue[str | None],
        out: asyncio.Queue[bytes | None],
        cancel: asyncio.Event,
    ) -> None:
        """
        Reads sentence strings from *sentences*, puts PCM bytes onto *out*.
        Must drain *sentences* and return cleanly when *cancel* is set.
        Responsible for maintaining its own warm WebSocket connection.
        """
        ...


@runtime_checkable
class AudioSink(Protocol):
    """Plays PCM audio from a queue."""

    async def run(
        self,
        audio: asyncio.Queue[bytes | None],
        cancel: asyncio.Event,
    ) -> None:
        """
        Reads PCM chunks from *audio* and plays them.
        Clears the queue and stops playback when *cancel* is set.
        """
        ...