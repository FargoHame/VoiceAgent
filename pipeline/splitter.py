from __future__ import annotations

import re
from typing import AsyncGenerator

# Only split on strong sentence boundaries followed by whitespace.
# Excludes abbreviations heuristic: don't split if preceded by a single capital (e.g. "U.S.")
_BOUNDARY = re.compile(r'(?<![A-Z])[.!?][)\]"\']*(?:\s|$)')


async def split_sentences(
    token_stream: AsyncGenerator[str, None],
    min_length: int = 200,
) -> AsyncGenerator[str, None]:
    buffer = ""

    async for token in token_stream:
        buffer += token

        if len(buffer) >= min_length:
            match = _BOUNDARY.search(buffer)
            if match:
                sentence = buffer[: match.end()].strip()
                remainder = buffer[match.end():]
                if sentence:
                    yield sentence
                buffer = remainder

    leftover = buffer.strip()
    if leftover:
        yield leftover