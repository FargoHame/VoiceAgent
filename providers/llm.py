"""
Groq LLM provider.

Implements the LanguageModel protocol.

Design constraints:
  - Exposes only `generate(prompt)` — an async generator of token strings.
  - Queue management is the caller's responsibility; this class owns no queues.
  - Cancellation: the caller simply stops iterating the generator.
  - Maintains conversation history internally so context is preserved across
    turns without the orchestrator needing to track it.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator

from groq import AsyncGroq

from config import Config

logger = logging.getLogger(__name__)

_MODEL = "llama-3.1-8b-instant"
_MAX_TOKENS = 200


class GroqLLM:
    """Stateful LLM client with rolling conversation history."""

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=Config.GROQ_API_KEY)
        self._history: list[dict[str, str]] = [
            {"role": "system", "content": Config.SYSTEM_PROMPT}
        ]

    async def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Yield token strings for *prompt*, streaming from the Groq API.

        The full response is appended to the internal conversation history
        only after the stream completes, so partial responses on interruption
        are not stored.

        Cancellation: stop iterating.  The underlying HTTP stream is abandoned
        gracefully by the Groq client on garbage collection.
        """
        self._history.append({"role": "user", "content": prompt})

        try:
            stream = await self._client.chat.completions.create(
                model=_MODEL,
                messages=self._history,
                stream=True,
                max_tokens=_MAX_TOKENS,
            )
        except Exception as exc:
            logger.error("LLM API error initiating stream: %s", exc)
            # Remove the user turn we just added so history stays consistent.
            self._history.pop()
            return

        accumulated = ""
        try:
            async for chunk in stream:
                token: str = chunk.choices[0].delta.content or ""
                if token:
                    accumulated += token
                    yield token
        except Exception as exc:
            logger.error("LLM stream error: %s", exc)
        finally:
            # Only commit to history if we got a meaningful response.
            if accumulated:
                self._history.append({"role": "assistant", "content": accumulated})
                logger.info("LLM response (%d chars)", len(accumulated))

    def reset_history(self) -> None:
        """Clear conversation history, keeping only the system prompt."""
        self._history = [self._history[0]]