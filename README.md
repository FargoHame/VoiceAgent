# Voice Agent

A low-latency streaming voice AI agent built with Deepgram (STT + TTS) and Groq (Llama 3.1). Speaks and listens in real time with barge-in interruption support.

## Architecture

```
Microphone → Deepgram STT → Groq LLM → Sentence Splitter → Deepgram TTS → Speaker
```

Each stage is decoupled and communicates only through queues. No component holds a reference to another — the orchestrator owns all wiring.

| Component | Role |
|---|---|
| `audio_io.py` | Hardware mic capture and speaker playback |
| `providers/stt.py` | Persistent Deepgram STT WebSocket, auto-reconnects |
| `providers/llm.py` | Groq streaming token generator, manages conversation history |
| `providers/tts.py` | Persistent Deepgram TTS WebSocket, sentence-level synthesis |
| `pipeline/splitter.py` | Pure async sentence splitter — buffers LLM tokens, yields complete sentences |
| `pipeline/interfaces.py` | Protocol definitions for all pipeline components |
| `core/orchestrator.py` | Wires all components, owns queues and events, manages turn lifecycle |

## Features

- **~300–600ms time-to-first-audio** after speech detected
- **Persistent WebSockets** — STT and TTS connections stay warm across turns, no per-turn handshake cost
- **Cascaded streaming** — TTS starts synthesising the first sentence while LLM is still generating the rest
- **Barge-in interruption** — user can speak mid-response; current playback stops instantly and queue is drained
- **Auto-reconnect** — both STT and TTS reconnect automatically on unexpected drops
- **Debounced STT** — rapid Deepgram `is_final` events within 300ms are merged into one turn
- **Thread-safe audio** — playback runs in a dedicated thread with a `queue.Queue` for zero dropped chunks

## Stack

- [Deepgram](https://deepgram.com) — STT and TTS via WebSocket streaming
- [Groq](https://groq.com) — Llama 3.1 8B inference
- [sounddevice](https://python-sounddevice.readthedocs.io) — cross-platform audio I/O
- Python `asyncio` — async pipeline throughout

## Setup

**1. Install dependencies**

```bash
pip install websockets groq sounddevice python-dotenv
```

On Linux:
```bash
sudo apt install portaudio19-dev
```

On macOS:
```bash
brew install portaudio
```

**2. Create a `.env` file**

```
GROQ_API_KEY=your_groq_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
SYSTEM_PROMPT=You are a concise, helpful voice assistant.
```

**3. Run**

```bash
python main.py
```

Press `Ctrl+C` to stop.

## Project Structure

```
voice_agent/
├── main.py
├── config.py
├── audio_io.py
├── pipeline/
│   ├── __init__.py
│   ├── interfaces.py
│   └── splitter.py
├── providers/
│   ├── llm.py
│   ├── stt.py
│   └── tts.py
└── core/
    └── orchestrator.py
```

## Configuration

All tunable values are in `config.py` and `providers/stt.py`:

| Setting | Default | Description |
|---|---|---|
| `AUDIO.CHUNK` | `1024` | Frames per audio buffer |
| `AUDIO.RATE` | `16000` | Sample rate in Hz |
| `endpointing` | `500ms` | How long Deepgram waits after speech ends before firing |
| `min_length` | `200` | Minimum chars before sentence splitter fires a TTS call |
| `max_tokens` | `200` | Max LLM response length |

## Latency Breakdown

| Stage | Typical latency |
|---|---|
| STT endpointing (after you stop speaking) | ~500ms |
| STT → LLM first token | ~0ms |
| LLM → sentence splitter → TTS send | ~0ms |
| TTS first audio chunk (Deepgram network) | ~300–600ms |
| **Total perceived** | **~800ms–1.1s** |
