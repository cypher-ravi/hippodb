"""Run a memory system on LoCoMo and report accuracy and tokens per answer.

uv run engram-bench --system rag --conversations 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from engram.bench import locomo
from engram.bench.answer import answer, judge
from engram.llm import LLM
from engram.memory import MemorySystem
from engram.tokens import count_tokens


@dataclass
class Result:
    sample_id: str
    category: int
    question: str
    gold: str
    generated: str
    correct: bool
    context_tokens: int
    retrieve_ms: float


def evaluate(
    system: MemorySystem,
    conversations: list[locomo.Conversation],
    answer_llm: LLM,
    judge_llm: LLM,
    include_adversarial: bool = False,
    on_result=None,
) -> list[Result]:
    results: list[Result] = []
    for conv in conversations:
        system.ingest(conv.turns)
        for q in conv.questions:
            if q.category == 5 and not include_adversarial:
                continue
            start = time.perf_counter()
            context = system.retrieve(q.question)
            retrieve_ms = (time.perf_counter() - start) * 1000
            generated = answer(answer_llm, q.question, context).text
            result = Result(
                sample_id=conv.sample_id,
                category=q.category,
                question=q.question,
                gold=q.answer,
                generated=generated,
                correct=judge(judge_llm, q.question, q.answer, generated),
                context_tokens=count_tokens(context),
                retrieve_ms=retrieve_ms,
            )
            results.append(result)
            if on_result:
                on_result(result)
    return results


def summarise(results: list[Result]) -> dict:
    by_cat: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_cat[locomo.CATEGORIES[r.category]].append(r)

    def stats(rs: list[Result]) -> dict:
        return {
            "n": len(rs),
            "accuracy": round(sum(r.correct for r in rs) / len(rs), 4),
            "mean_context_tokens": round(statistics.mean(r.context_tokens for r in rs), 1),
        }

    latencies = sorted(r.retrieve_ms for r in results)
    return {
        "overall": stats(results),
        "by_category": {cat: stats(rs) for cat, rs in sorted(by_cat.items())},
        "retrieve_ms_p50": round(statistics.median(latencies), 2),
        "retrieve_ms_p99": round(latencies[int(0.99 * (len(latencies) - 1))], 2),
    }


def print_summary(name: str, summary: dict) -> None:
    print(f"\n{name}")
    print(f"{'category':<14}{'n':>6}{'accuracy':>10}{'ctx tokens':>12}")
    rows = [*summary["by_category"].items(), ("OVERALL", summary["overall"])]
    for cat, s in rows:
        print(f"{cat:<14}{s['n']:>6}{s['accuracy']:>10.1%}{s['mean_context_tokens']:>12.0f}")
    print(f"retrieve p50 {summary['retrieve_ms_p50']} ms, p99 {summary['retrieve_ms_p99']} ms")


def build_system(name: str, k: int) -> MemorySystem:
    if name == "rag":
        from engram.baselines.vector_rag import VectorRAG
        from engram.embed import FastEmbedder

        return VectorRAG(FastEmbedder(), k=k)
    raise SystemExit(f"Unknown system: {name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", default="rag", choices=["rag"])
    parser.add_argument("--conversations", type=int, default=2, help="first N of 10")
    parser.add_argument("--k", type=int, default=10, help="chunks retrieved per question")
    parser.add_argument("--model", help="answering/judging model (default from env)")
    parser.add_argument("--include-adversarial", action="store_true")
    parser.add_argument("--data", type=Path, default=locomo.DEFAULT_PATH)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    from engram.llm import from_env

    _load_dotenv()
    llm = from_env(args.model)
    conversations = locomo.load(locomo.download(args.data))[: args.conversations]
    system = build_system(args.system, args.k)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_path = args.out / f"{system.name}-{stamp}.jsonl"
    with run_path.open("w") as f:

        def on_result(r: Result) -> None:
            f.write(json.dumps(asdict(r)) + "\n")
            mark = "✓" if r.correct else "✗"
            print(f"{mark} [{locomo.CATEGORIES[r.category]}] {r.question}")

        results = evaluate(system, conversations, llm, llm, args.include_adversarial, on_result)

    summary = summarise(results) | {
        "system": system.name,
        "model": llm.model,
        "conversations": [c.sample_id for c in conversations],
    }
    run_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(f"{system.name} · {llm.model}", summary)
    print(f"\nPer-question results: {run_path}")


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env support so API keys don't have to be exported by hand."""
    import os

    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#") and value.strip():
            os.environ.setdefault(key.strip(), value.strip())


if __name__ == "__main__":
    main()
