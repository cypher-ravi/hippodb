"""Download the benchmark datasets into data/.

uv run python -m hippodb.bench.download
"""

from hippodb.bench import locomo

if __name__ == "__main__":
    path = locomo.download()
    conversations = locomo.load(path)
    n_q = sum(len(c.questions) for c in conversations)
    print(f"LoCoMo: {len(conversations)} conversations, {n_q} questions -> {path}")
