import os
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

class ConfigError(Exception):
    pass

def get_env_var(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {var_name}")
    return value

@dataclass(frozen=True)
class AudioConfig:
    FORMAT: int = 8  # pyaudio.paInt16
    CHANNELS: int = 1
    RATE: int = 16000
    CHUNK: int = 1024

class Config:
    GROQ_API_KEY: str = get_env_var("GROQ_API_KEY")
    DEEPGRAM_API_KEY: str = get_env_var("DEEPGRAM_API_KEY")
    SYSTEM_PROMPT: str = os.getenv("SYSTEM_PROMPT", "You are a concise AI.")
    AUDIO = AudioConfig()
