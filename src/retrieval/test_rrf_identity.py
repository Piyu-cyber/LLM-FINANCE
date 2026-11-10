"""test_rrf_identity.py — Unit tests for RRF identifier stability and model reuse."""

import sys
import math
from unittest.mock import MagicMock

# Fast unit test stubs so tests don't load heavyweight torch/CUDA binaries on CPU
for _mod in ("torch", "sentence_transformers", "transformers", "rank_bm25"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            _mock = MagicMock()
            if _mod == "torch":
                _mock.cuda.is_available.return_value = False
            sys.modules[_mod] = _mock

from rrf_retriever import (
    get_chunk_uid,
    reciprocal_rank_fusion,
    retrieve,
)


def test_get_chunk_uid_formatting():
    chunk1 = {"example_id": "UNP/2009/page_34.pdf-1", "chunk_id": "table_1", "text": "foo"}
    assert get_chunk_uid(chunk1) == "UNP/2009/page_34.pdf-1#table_1"

    chunk2 = {"id": "chunk_99", "text": "bar"}
    assert get_chunk_uid(chunk2) == "chunk_99"


def test_rrf_ranking_stability_across_reordered_corpus():
    # Two chunks
    c1 = {"example_id": "docA", "chunk_id": "c1", "text": "interest expense"}
    c2 = {"example_id": "docB", "chunk_id": "c2", "text": "revenue growth"}

    uid1 = get_chunk_uid(c1)
    uid2 = get_chunk_uid(c2)

    # In BM25, c1 is rank 1, c2 is rank 2
    bm25_res = [(uid1, 10.0), (uid2, 5.0)]
    # In Dense, c1 is rank 2, c2 is rank 1
    dense_res = [(uid2, 0.9), (uid1, 0.7)]

    # Fused scores should be independent of corpus list order
    fused1 = reciprocal_rank_fusion(bm25_res, dense_res, k=60)
    score1 = dict(fused1)

    # Now reverse order of inputs
    fused2 = reciprocal_rank_fusion(list(reversed(bm25_res)), list(reversed(dense_res)), k=60)
    score2 = dict(fused2)

    assert set(score1.keys()) == {uid1, uid2}
    assert math.isclose(score1[uid1], score2[uid1])
    assert math.isclose(score1[uid2], score2[uid2])


def test_rrf_handles_duplicate_chunk_uids():
    c1 = {"example_id": "docA", "chunk_id": "c1", "text": "first"}
    c1_dup = {"example_id": "docA", "chunk_id": "c1", "text": "first duplicate"}
    c2 = {"example_id": "docB", "chunk_id": "c2", "text": "second"}

    chunks = [c1, c1_dup, c2]

    uid1 = get_chunk_uid(c1)
    uid2 = get_chunk_uid(c2)

    bm25_res = [(uid1, 10.0), (uid2, 5.0)]
    dense_res = [(uid1, 0.8), (uid2, 0.6)]

    fused = reciprocal_rank_fusion(bm25_res, dense_res, k=60)
    # Result should have exactly 2 unique UIDs, no duplicates
    assert len(fused) == 2
    assert fused[0][0] == uid1


def test_retrieve_uses_injected_reranker():
    c1 = {"example_id": "docA", "chunk_id": "c1", "text": "test document"}
    chunks = [c1]

    mock_bm25 = MagicMock()
    mock_bm25.get_scores.return_value = [5.0]

    mock_dense_model = MagicMock()
    mock_dense_model.encode.return_value = [[0.1, 0.2]]

    mock_embeddings = MagicMock()
    mock_embeddings.__matmul__ = MagicMock(return_value=[0.8])

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [dict(c1, rerank_score=0.99)]

    res = retrieve(
        query="What is test?",
        chunks=chunks,
        bm25=mock_bm25,
        dense_model=mock_dense_model,
        embeddings=mock_embeddings,
        top_k=1,
        pool_size=1,
        reranker_type="cross_encoder",
        reranker=mock_reranker,  # INJECTED
        apply_metadata_boosting=False,
    )

    # Injected reranker must be called
    mock_reranker.rerank.assert_called_once()
    assert len(res) == 1
    assert res[0]["rerank_score"] == 0.99
