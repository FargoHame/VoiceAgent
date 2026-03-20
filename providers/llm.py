import logging
from typing import AsyncGenerator, List, Dict
from groq import AsyncGroq
from config import Config

logger = logging.getLogger(__name__)

class GroqLLM:
    
    def __init__(self) -> None:
        self.client = AsyncGroq(api_key=Config.GROQ_API_KEY)
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": Config.SYSTEM_PROMPT}]

    async def generate(self, prompt: str) -> AsyncGenerator[str, None]:
        self.messages.append({"role": "user", "content": prompt})
        
        try:
            response = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=self.messages,
                stream=True,
                max_tokens=150
            )
            
            full_response = ""
            async for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content
            
            # Save assistant context for memory
            self.messages.append({"role": "assistant", "content": full_response})
            logger.info(f"AI Responded: {full_response}")
            
        except Exception as e:
            logger.error(f"LLM Generation API error: {e}")
            yield "Sorry, I encountered an error."
