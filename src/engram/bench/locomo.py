"""Download and load the LoCoMo long-conversation memory benchmark.

LoCoMo (Maharana et al., 2024) has 10 long two-person conversations, each split
into dated sessions, plus question/answer pairs about them.
https://github.com/snap-research/locomo
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_PATH = Path("data/locomo10.json")

# Category ids as used in the LoCoMo repo. Category 5 (adversarial) questions have
# no answer in the conversation; most published results (e.g. Mem0) leave them out.
CATEGORIES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}


@dataclass(frozen=True)
class Turn:
    dia_id: str  # e.g. "D3:12" = session 3, turn 12
    session: int
    session_date: str  # e.g. "1:56 pm on 8 May, 2023"
    speaker: str
    text: str


@dataclass(frozen=True)
class Question:
    question: str
    answer: str
    category: int
    evidence: tuple[str, ...]  # dia_ids of the turns that hold the answer


@dataclass(frozen=True)
class Conversation:
    sample_id: str
    speakers: tuple[str, str]
    turns: list[Turn]
    questions: list[Question]


def download(path: Path = DEFAULT_PATH) -> Path:
    """Download locomo10.json to `path` unless it is already there."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LOCOMO_URL, path)
    return path


def load(path: Path = DEFAULT_PATH) -> list[Conversation]:
    with path.open() as f:
        raw = json.load(f)
    return [_parse(sample) for sample in raw]


def _parse(sample: dict) -> Conversation:
    conv = sample["conversation"]
    turns: list[Turn] = []
    session = 1
    while f"session_{session}" in conv:
        date = conv.get(f"session_{session}_date_time", "")
        for t in conv[f"session_{session}"]:
            text = t["text"]
            # Some turns share a photo; the dataset gives a caption for it.
            if t.get("blip_caption"):
                text = f"{text} [shares a photo of {t['blip_caption']}]"
            turns.append(Turn(t["dia_id"], session, date, t["speaker"], text))
        session += 1

    questions = [
        Question(
            question=q["question"],
            # Adversarial questions have no `answer`, only an `adversarial_answer` trap.
            answer=str(q.get("answer", "")),
            category=q["category"],
            evidence=tuple(q.get("evidence", [])),
        )
        for q in sample["qa"]
    ]
    return Conversation(
        sample_id=sample["sample_id"],
        speakers=(conv["speaker_a"], conv["speaker_b"]),
        turns=turns,
        questions=questions,
    )
