from engram.baselines.vector_rag import VectorRAG
from engram.bench import locomo
from engram.bench.answer import judge
from engram.bench.run import evaluate, summarise
from engram.llm import CachedLLM, Completion


def test_evaluate_skips_adversarial_and_scores(locomo_file, embedder, llm):
    convs = locomo.load(locomo_file)
    results = evaluate(VectorRAG(embedder, k=1), convs, llm, llm)
    assert [r.category for r in results] == [4, 2]
    assert [r.correct for r in results] == [True, False]
    assert all(r.context_tokens > 0 for r in results)

    summary = summarise(results)
    assert summary["overall"]["n"] == 2
    assert summary["overall"]["accuracy"] == 0.5
    assert summary["by_category"]["single_hop"]["accuracy"] == 1.0


def test_include_adversarial(locomo_file, embedder, llm):
    convs = locomo.load(locomo_file)
    results = evaluate(VectorRAG(embedder), convs, llm, llm, include_adversarial=True)
    assert len(results) == 3


class FixedLLM:
    model = "fixed"

    def __init__(self, text):
        self.text = text

    def complete(self, system, prompt, max_tokens=200):
        return Completion(self.text, 1, 1)


def test_judge_does_not_read_incorrect_as_correct():
    assert judge(FixedLLM("CORRECT"), "q", "a", "a")
    assert judge(FixedLLM("correct."), "q", "a", "a")
    assert not judge(FixedLLM("INCORRECT"), "q", "a", "b")
    assert not judge(FixedLLM("WRONG"), "q", "a", "b")


def test_cached_llm_calls_inner_once(tmp_path, llm):
    cached = CachedLLM(llm, tmp_path / "cache.sqlite")
    first = cached.complete("sys", "Memories: Bruno")
    second = cached.complete("sys", "Memories: Bruno")
    assert first == second
    assert llm.calls == 1
