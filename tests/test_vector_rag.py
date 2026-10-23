from engram.baselines.vector_rag import VectorRAG
from engram.bench import locomo


def test_retrieves_relevant_turn_with_date(locomo_file, embedder):
    [conv] = locomo.load(locomo_file)
    rag = VectorRAG(embedder, k=1)
    rag.ingest(conv.turns)
    context = rag.retrieve("What is the dog called that Asha adopted?")
    assert context == "[10:00 am on 1 May, 2023] Asha: I adopted a dog called Bruno."


def test_k_larger_than_memory_returns_everything_in_order(locomo_file, embedder):
    [conv] = locomo.load(locomo_file)
    rag = VectorRAG(embedder, k=50)
    rag.ingest(conv.turns)
    lines = rag.retrieve("guitar").splitlines()
    assert len(lines) == 3
    assert "Bruno" in lines[0] and "guitar" in lines[1]


def test_empty_memory_returns_empty_context(embedder):
    assert VectorRAG(embedder).retrieve("anything") == ""
