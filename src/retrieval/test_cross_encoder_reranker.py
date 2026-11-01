"""test_cross_encoder_reranker.py — Unit tests for Phase 1 Cross-Encoder reranker.

Run with:
    $env:PYTHONPATH="src;src/retrieval;src/verification;src/eval"
    python -m pytest src/retrieval/test_cross_encoder_reranker.py -v
"""

import sys
import os
import math
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

# Fast unit test stubs so tests don't load heavyweight torch/CUDA binaries on CPU
if "torch" not in sys.modules:
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    sys.modules["torch"] = torch_mock

if "sentence_transformers" not in sys.modules:
    st_mock = MagicMock()
    sys.modules["sentence_transformers"] = st_mock

from cross_encoder_reranker import CrossEncoderReranker


def test_cross_encoder_rerank_logic():
    reranker = CrossEncoderReranker(model_name="dummy-model")
    # Mock predict method on the reranker's underlying model
    reranker.model.predict.return_value = [0.2, 0.9, 0.5]

    candidates = [
        {"chunk_id": "c1", "text": "candidate 1"},
        {"chunk_id": "c2", "text": "candidate 2"},
        {"chunk_id": "c3", "text": "candidate 3"},
    ]

    query = "What is the revenue?"
    top_k = 2
    results = reranker.rerank(query, candidates, top_k=top_k)

    assert len(results) == 2
    # Highest score was 0.9 for candidate 2
    assert results[0]["chunk_id"] == "c2"
    assert math.isclose(results[0]["rerank_score"], 0.9)
    # Second highest score was 0.5 for candidate 3
    assert results[1]["chunk_id"] == "c3"
    assert math.isclose(results[1]["rerank_score"], 0.5)


def test_cross_encoder_empty_candidates():
    reranker = CrossEncoderReranker(model_name="dummy-model")
    results = reranker.rerank("Query", [], top_k=5)
    assert results == []


def test_cross_encoder_preserves_original_chunk_metadata():
    reranker = CrossEncoderReranker(model_name="dummy-model")
    reranker.model.predict.return_value = [0.85]

    candidate = {
        "example_id": "UNP/2009/page_34.pdf-1",
        "chunk_id": "table_1",
        "source": "table",
        "text": "interest expense",
        "parent_id": "UNP/2009/page_34.pdf-1",
    }

    results = reranker.rerank("What is interest expense?", [candidate], top_k=1)
    assert len(results) == 1
    assert results[0]["example_id"] == "UNP/2009/page_34.pdf-1"
    assert results[0]["chunk_id"] == "table_1"
    assert results[0]["source"] == "table"
    assert math.isclose(results[0]["rerank_score"], 0.85)
