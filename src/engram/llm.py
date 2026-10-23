"""A thin LLM client with an on-disk cache.

Answering and judging run the same prompts again and again while you tune a
system, so every completion is cached in SQLite keyed by (model, prompt).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
}
CACHE_PATH = Path(".cache/llm.sqlite")


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


class LLM(Protocol):
    model: str

    def complete(self, system: str, prompt: str, max_tokens: int = 200) -> Completion: ...


class AnthropicLLM:
    def __init__(self, model: str):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic()

    def complete(self, system: str, prompt: str, max_tokens: int = 200) -> Completion:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(text.strip(), resp.usage.input_tokens, resp.usage.output_tokens)


class OpenAILLM:
    def __init__(self, model: str):
        import openai

        self.model = model
        self._client = openai.OpenAI()

    def complete(self, system: str, prompt: str, max_tokens: int = 200) -> Completion:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        usage = resp.usage
        return Completion(
            (resp.choices[0].message.content or "").strip(),
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )


class CachedLLM:
    """Wraps any LLM; identical (model, system, prompt) calls hit the cache."""

    def __init__(self, inner: LLM, path: Path = CACHE_PATH):
        self.inner = inner
        self.model = inner.model
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def complete(self, system: str, prompt: str, max_tokens: int = 200) -> Completion:
        key = hashlib.sha256(
            json.dumps([self.model, system, prompt, max_tokens]).encode()
        ).hexdigest()
        row = self._db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            return Completion(**json.loads(row[0]))
        result = self.inner.complete(system, prompt, max_tokens)
        self._db.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(result.__dict__))
        )
        self._db.commit()
        return result


def from_env(model: str | None = None) -> LLM:
    """Pick a provider from ENGRAM_LLM_PROVIDER or whichever API key is set."""
    provider = os.environ.get("ENGRAM_LLM_PROVIDER")
    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise SystemExit("Set ANTHROPIC_API_KEY or OPENAI_API_KEY (see .env.example).")
    model = model or os.environ.get("ENGRAM_LLM_MODEL") or DEFAULT_MODELS[provider]
    inner: LLM = AnthropicLLM(model) if provider == "anthropic" else OpenAILLM(model)
    return CachedLLM(inner)
