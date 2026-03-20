import os
import sys

if sys.platform == 'win32':
    os.add_dll_directory(os.getcwd())


import asyncio
import logging
from core.orchestrator import AgentOrchestrator

# Standardized logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

async def main() -> None:
    agent = AgentOrchestrator()
    try:
        await agent.run()
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping agent...")
    finally:
        agent.shutdown()

if __name__ == "__main__":
    # Ensures a clean event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
