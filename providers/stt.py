import json
import websockets
import asyncio
import logging
from config import Config

logger = logging.getLogger(__name__)

class DeepgramSTT:
    def __init__(self) -> None:
        # Note: 'endpointing' is often better handled by the provider 
        # but 300ms is quite aggressive for some users.
        self.url = "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=16000&channels=1&endpointing=300"
        # Use additional_headers for newer websockets versions
        self.headers = {"Authorization": f"Token {Config.DEEPGRAM_API_KEY}"}

    async def run(self, mic_queue: asyncio.Queue, text_queue: asyncio.Queue) -> None:
        try:
            # ping_interval helps keep the connection alive during silence
            async with websockets.connect(
                self.url, 
                additional_headers=self.headers,
                ping_interval=10
            ) as ws:
                logger.info("STT WebSocket Connected.")
                
                async def sender():
                    try:
                        while True:
                            data = await mic_queue.get()
                            if data is None:  # Sentinel value for shutdown
                                await ws.send(json.dumps({"type": "CloseStream"}))
                                break
                            await ws.send(data)
                            mic_queue.task_done()
                    except Exception as e:
                        logger.error(f"STT Sender Error: {e}")

                async def receiver():
                    try:
                        async for message in ws:
                            res = json.loads(message)
                            # Deepgram marks the final transcript for a segment with is_final
                            is_final = res.get("is_final", False)
                            channel = res.get("channel", {})
                            alternatives = channel.get("alternatives", [{}])
                            transcript = alternatives[0].get("transcript", "")
                            
                            if transcript and is_final:
                                logger.info(f"User Transcribed: {transcript}")
                                await text_queue.put(transcript)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("STT WebSocket closed.")
                    except Exception as e:
                        logger.error(f"STT Receiver Error: {e}")

                # Using gather to run both tasks concurrently until one fails or finishes
                await asyncio.gather(sender(), receiver())
                
        except Exception as e:
            logger.error(f"STT Connection error: {e}")