import json
import zlib

import numpy as np
import pytest

from engram.embed import normalise
from engram.llm import Completion

SAMPLE = {
    "sample_id": "conv-test",
    "conversation": {
        "speaker_a": "Asha",
        "speaker_b": "Ben",
        "session_1_date_time": "10:00 am on 1 May, 2023",
        "session_1": [
            {"speaker": "Asha", "dia_id": "D1:1", "text": "I adopted a dog called Bruno."},
            {"speaker": "Ben", "dia_id": "D1:2", "text": "Lovely! I started learning guitar."},
        ],
        "session_2_date_time": "4:00 pm on 9 June, 2023",
        "session_2": [
            {
                "speaker": "Asha",
                "dia_id": "D2:1",
                "text": "Look at this.",
                "blip_caption": "a dog on a beach",
            },
        ],
    },
    "qa": [
        {
            "question": "What is Asha's dog called?",
            "answer": "Bruno",
            "evidence": ["D1:1"],
            "category": 4,
        },
        {
            "question": "What year did Ben start guitar?",
            "answer": 2023,
            "evidence": ["D1:2"],
            "category": 2,
        },
        {
            "question": "What is Ben's cat called?",
            "evidence": [],
            "category": 5,
            "adversarial_answer": "Max",
        },
    ],
}


@pytest.fixture
def locomo_file(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps([SAMPLE]))
    return path


class BagOfWordsEmbedder:
    """Deterministic stand-in for a real model: hashed word counts."""

    model_name = "bow-test"

    def embed(self, texts):
        vecs = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().replace("?", " ").replace(".", " ").split():
                vecs[i, zlib.crc32(word.encode()) % 64] += 1
        return normalise(vecs)


class ScriptedLLM:
    """Answers with the first gold word found in the prompt; judges by substring."""

    model = "scripted"

    def __init__(self):
        self.calls = 0

    def complete(self, system, prompt, max_tokens=200):
        self.calls += 1
        if prompt.startswith("Question:"):  # judge
            gold = prompt.split("Gold answer: ")[1].split("\n")[0]
            generated = prompt.split("Generated answer: ")[1]
            return Completion("CORRECT" if gold in generated else "WRONG", 10, 1)
        return Completion("Bruno" if "Bruno" in prompt else "Not mentioned", 10, 1)


@pytest.fixture(autouse=True)
def offline_token_count(monkeypatch):
    """tiktoken downloads its vocabulary on first use; keep tests offline."""
    monkeypatch.setattr("engram.bench.run.count_tokens", lambda text: len(text.split()))


@pytest.fixture
def embedder():
    return BagOfWordsEmbedder()


@pytest.fixture
def llm():
    return ScriptedLLM()
