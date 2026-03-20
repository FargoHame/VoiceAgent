import sounddevice as sd
import numpy as np
import asyncio
import logging
from config import Config

logger = logging.getLogger(__name__)

class AudioManager:
    """Uses sounddevice for robust, deployment-ready audio I/O."""
    
    def __init__(self) -> None:
        self.stream_input = None
        self.stream_output = None
        self.interrupt_event = asyncio.Event()
        self._is_playing = False  # Track playback state internally

    @property
    def is_playing(self) -> bool:
        """Staff-fix: Property requested by core.orchestrator."""
        return self._is_playing

    def start_microphone(self, queue: asyncio.Queue) -> None:
        """Callback-based recording. More stable than threaded loops."""
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Microphone status: {status}")
            # Directly convert the raw buffer to bytes for stability
            queue.put_nowait(bytes(indata))

        try:
            self.stream_input = sd.RawInputStream(
                samplerate=Config.AUDIO.RATE,
                blocksize=Config.AUDIO.CHUNK,
                device=None,
                channels=Config.AUDIO.CHANNELS,
                dtype='int16',
                callback=callback
            )
            with self.stream_input:
                logger.info("Microphone hardware active via sounddevice.")
                while self.stream_input.active:
                    sd.sleep(100)
        except Exception as e:
            logger.error(f"Microphone hardware error: {e}")

    async def play_audio(self, audio_queue: asyncio.Queue) -> None:
        """Non-blocking playback using a generator-based output stream."""
        try:
            self.stream_output = sd.RawOutputStream(
                samplerate=Config.AUDIO.RATE,
                blocksize=Config.AUDIO.CHUNK,
                device=None,
                channels=Config.AUDIO.CHANNELS,
                dtype='int16'
            )
            self.stream_output.start()
            logger.info("Speaker hardware ready via sounddevice.")

            while True:
                if self.interrupt_event.is_set():
                    while not audio_queue.empty():
                        audio_queue.get_nowait()
                        audio_queue.task_done()
                    self.interrupt_event.clear()
                    self._is_playing = False
                    continue

                chunk = await audio_queue.get()
                if chunk is None: break
                
                # Update status for orchestrator
                self._is_playing = True
                self.stream_output.write(chunk)
                audio_queue.task_done()
                
                # If we've finished the current burst of audio, set playing to False
                if audio_queue.empty():
                    self._is_playing = False

        except Exception as e:
            logger.error(f"Playback error: {e}")
            self._is_playing = False
        finally:
            self._is_playing = False
            if self.stream_output:
                self.stream_output.stop()
                self.stream_output.close()

    def interrupt(self) -> None:
        self.interrupt_event.set()

    def shutdown(self) -> None:
        if self.stream_input: self.stream_input.stop()
        if self.stream_output: self.stream_output.stop()
        logger.info("Audio system offline.")