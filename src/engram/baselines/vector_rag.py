"""Plain vector RAG: the baseline Engram has to beat.

Each conversation turn is one chunk, stamped with its session date and speaker.
Retrieval embeds the question and returns the top-k turns by cosine similarity,
in conversation order.
"""

from __future__ import annotations

import numpy as np

from engram.bench.locomo import Turn
from engram.embed import Embedder


def format_turn(turn: Turn) -> str:
    return f"[{turn.session_date}] {turn.speaker}: {turn.text}"


class VectorRAG:
    def __init__(self, embedder: Embedder, k: int = 10):
        self.name = f"vector_rag_k{k}"
        self.embedder = embedder
        self.k = k
        self._chunks: list[str] = []
        self._vectors = np.zeros((0, 0), dtype=np.float32)

    def ingest(self, turns: list[Turn]) -> None:
        self._chunks = [format_turn(t) for t in turns]
        self._vectors = self.embedder.embed(self._chunks)

    def retrieve(self, question: str) -> str:
        if not self._chunks:
            return ""
        q = self.embedder.embed([question])[0]
        scores = self._vectors @ q
        k = min(self.k, len(self._chunks))
        top = np.argpartition(-scores, k - 1)[:k]
        # Keep conversation order so the LLM reads events in the order they happened.
        return "\n".join(self._chunks[i] for i in sorted(top))
