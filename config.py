"""
Application configuration.

All environment variables are validated at import time so the process
fails fast with a clear message rather than crashing mid-call.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    pass


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class AudioConfig:
    CHANNELS: int = 1
    RATE: int = 16000
    CHUNK: int = 1024  # frames per buffer (~64 ms at 16 kHz)


class Config:
    GROQ_API_KEY: str = _require("GROQ_API_KEY")
    DEEPGRAM_API_KEY: str = _require("DEEPGRAM_API_KEY")
    SYSTEM_PROMPT: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are a concise, helpful voice assistant. Keep responses brief.",
    )
    AUDIO: AudioConfig = AudioConfig()