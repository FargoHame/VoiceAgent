"""Entry point."""

from __future__ import annotations

import asyncio
import logging
import sys

if sys.platform == "win32":
    import os
    os.add_dll_directory(os.getcwd())

from core.orchestrator import AgentOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


async def _run() -> None:
    agent = AgentOrchestrator()

    try:
        await agent.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        agent.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass