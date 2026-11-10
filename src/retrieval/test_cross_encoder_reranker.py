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
    try:
        import torch
    except ImportError:
        torch_mock = MagicMock()
        torch_mock.cuda.is_available.return_value = False
        sys.modules["torch"] = torch_mock

if "sentence_transformers" not in sys.modules:
    try:
        import sentence_transformers
    except ImportError:
        st_mock = MagicMock()
        sys.modules["sentence_transformers"] = st_mock

from cross_encoder_reranker import CrossEncoderReranker


def test_cross_encoder_rerank_logic():
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.2, 0.9, 0.5]
    reranker = CrossEncoderReranker(model_name="dummy-model", model=mock_model)

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
    mock_model = MagicMock()
    reranker = CrossEncoderReranker(model_name="dummy-model", model=mock_model)
    results = reranker.rerank("Query", [], top_k=5)
    assert results == []


def test_cross_encoder_preserves_original_chunk_metadata():
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.85]
    reranker = CrossEncoderReranker(model_name="dummy-model", model=mock_model)

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


def test_cross_encoder_singleton_and_injection():
    from unittest.mock import patch
    from cross_encoder_reranker import get_cross_encoder, reset_cross_encoder_cache

    reset_cross_encoder_cache()
    mock_model = MagicMock()
    # Test dependency injection
    injected_reranker = CrossEncoderReranker(model_name="mock-model", model=mock_model)
    assert injected_reranker.model is mock_model

    # Test singleton caching with mocked CrossEncoder to prevent offline download
    with patch("cross_encoder_reranker.CrossEncoder"):
        r1 = get_cross_encoder(model_name="dummy-singleton")
        r2 = get_cross_encoder(model_name="dummy-singleton")
        assert r1 is r2  # Exact same instance reused, model not re-loaded

        reset_cross_encoder_cache()
        r3 = get_cross_encoder(model_name="dummy-singleton")
        assert r3 is not r1  # Cache reset creates fresh instance
