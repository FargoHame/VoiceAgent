import json
import websockets
import asyncio
import logging
from typing import AsyncGenerator
from config import Config

logger = logging.getLogger(__name__)

class DeepgramTTS:
    def __init__(self) -> None:
        # Standardize URL to use the aura-asteria model and 16k linear16 encoding
        self.url = "wss://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=16000"
        self.headers = {"Authorization": f"Token {Config.DEEPGRAM_API_KEY}"}

    async def generate_audio(self, text_stream: AsyncGenerator[str, None], audio_queue: asyncio.Queue) -> None:
        """
        Takes a stream of text (from the LLM) and puts audio bytes into the audio_queue.
        """
        try:
            # FIX: Use 'additional_headers' for compatibility with modern websockets library
            async with websockets.connect(
                self.url, 
                additional_headers=self.headers,
                ping_interval=10 
            ) as ws:
                
                async def sender() -> None:
                    try:
                        async for text_chunk in text_stream:
                            if text_chunk:
                                # Deepgram Aura expects a JSON with a "text" field
                                msg = {"type": "Speak", "text": text_chunk}
                                await ws.send(json.dumps(msg))
                        
                        # Signal Deepgram that we are done sending text
                        await ws.send(json.dumps({"type": "Flush"}))
                    except Exception as e:
                        logger.error(f"TTS Sender Error: {e}")

                async def receiver() -> None:
                    try:
                        async for message in ws:
                            if isinstance(message, bytes):
                                # Feed the raw PCM bytes to the audio player
                                await audio_queue.put(message)
                            else:
                                resp = json.loads(message)
                                # When we see 'Flushed', the current sentence is done
                                if resp.get("type") == "Flushed":
                                    break
                    except websockets.exceptions.ConnectionClosed:
                        logger.debug("TTS WebSocket closed normally.")
                    except Exception as e:
                        logger.error(f"TTS Receiver Error: {e}")

                # Run sender and receiver concurrently
                await asyncio.gather(sender(), receiver())
                
        except Exception as e:
            logger.error(f"TTS Connection error: {e}")

    def shutdown(self):
        logger.info("TTS Provider shutting down.")