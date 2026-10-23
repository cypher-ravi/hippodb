# engram

A memory engine for AI agents. Engram stores **entities** and the **facts** that link
them, each with the time it became true and the source it came from. It recalls by
association (start from the entities a question mentions, spread to their neighbours),
closes facts that stop being true instead of overwriting them, and publishes
benchmarks against plain vector RAG.

> Status: weekend 1 of 14. The benchmark harness and the vector-RAG baseline are in;
> the graph prototype comes next.

## Plan

1. **Python prototype** that must beat vector RAG on LoCoMo (Gate 1).
2. **C++ server**: in-memory graph, write-ahead log, RESP protocol (works with
   `redis-cli`), HNSW entry points, spreading-activation `RECALL`.
3. **Python SDK + MCP server**: `remember()`, `recall()`, `forget()`.
4. **Benchmarks**: LoCoMo and LongMemEval vs vector RAG and Mem0, with accuracy,
   tokens per answer and recall latency.

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync                                   # install
uv run python -m engram.bench.download    # fetch LoCoMo into data/
cp .env.example .env                      # add ANTHROPIC_API_KEY or OPENAI_API_KEY
uv run engram-bench --system rag --conversations 2
```

The run prints accuracy by question category and the mean number of context tokens
given to the answering model, and writes per-question results to `results/`.
LLM calls are cached in `.cache/llm.sqlite`, so re-running is free.

Embeddings run locally (`BAAI/bge-small-en-v1.5` via fastembed, about 70 MB, downloaded
on first run). The default answering and judging model is `claude-haiku-5-5`
(or `gpt-4o-mini` with an OpenAI key); override with `--model`.

## Development

```bash
uv run pytest        # tests run offline, no API key needed
uv run ruff check .
uv run ruff format .
```

## Layout

```
src/engram/
  memory.py               MemorySystem interface every system implements
  baselines/vector_rag.py the baseline: one chunk per turn, top-k cosine
  bench/locomo.py         LoCoMo download + loader
  bench/answer.py         shared answer and judge prompts
  bench/run.py            the engram-bench CLI
  embed.py, llm.py        local embeddings; Anthropic/OpenAI client with cache
```

## License

MIT
