"""The interface every memory system implements, so the benchmark can compare them."""

from __future__ import annotations

from typing import Protocol

from engram.bench.locomo import Turn


class MemorySystem(Protocol):
    name: str

    def ingest(self, turns: list[Turn]) -> None:
        """Store a whole conversation. Called once per conversation, on a fresh system."""
        ...

    def retrieve(self, question: str) -> str:
        """Return the context text that will be given to the answering LLM."""
        ...
