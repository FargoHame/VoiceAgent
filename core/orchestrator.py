import asyncio
import threading
import logging
from audio_io import AudioManager
from providers.stt import DeepgramSTT
from providers.llm import GroqLLM
from providers.tts import DeepgramTTS

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    
    def __init__(self) -> None:
        self.audio = AudioManager()
        self.stt = DeepgramSTT()
        self.llm = GroqLLM()
        self.tts = DeepgramTTS()
        
        # Async queues decouple the pipeline stages
        self.mic_queue: asyncio.Queue = asyncio.Queue()
        self.text_queue: asyncio.Queue = asyncio.Queue()
        self.speaker_queue: asyncio.Queue = asyncio.Queue()

    async def brain_loop(self) -> None:
        try:
            while True:
                user_text = await self.text_queue.get()
                
                # Echo Cancellation / Interruption Logic
                if self.audio.is_playing and not self.speaker_queue.empty():
                    self.audio.interrupt()
                
                # 1. Start LLM Generation
                llm_stream = self.llm.generate(user_text)
                
                # 2. Pipe to TTS
                await self.tts.generate_audio(llm_stream, self.speaker_queue)                
                self.text_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Brain loop task cancelled.")

    async def run(self) -> None:
        # Microphone runs in a standard thread to avoid blocking asyncio
        mic_thread = threading.Thread(
            target=self.audio.start_microphone, 
            args=(self.mic_queue,), 
            daemon=True
        )
        mic_thread.start()
        
        # Run async pipeline components
        self.tasks = [
            asyncio.create_task(self.stt.run(self.mic_queue, self.text_queue)),
            asyncio.create_task(self.audio.play_audio(self.speaker_queue)),
            asyncio.create_task(self.brain_loop())
        ]
        
        logger.info("Agent is initialized and listening...")
        await asyncio.gather(*self.tasks)

    def shutdown(self) -> None:
        logger.info("Initiating shutdown sequence...")
        if hasattr(self, 'tasks'):
            for task in self.tasks:
                task.cancel()
        
        # Send poison pill to speaker queue
        try:
            self.speaker_queue.put_nowait(None)
        except Exception:
            pass
            
        self.audio.shutdown()
        logger.info("Shutdown complete.")
