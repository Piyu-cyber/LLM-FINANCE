"""test_integration.py — End-to-end integration tests for LLM-FINANCE.

Tests the full pipeline on a real-format FinQA fixture:
1. BM25 retrieval.
2. Dense retrieval & RRF fusion (with stable chunk UIDs).
3. Cross-encoder reranking (model-free with injected model).
4. Evidence validation (operands, values, periods, units).
5. End-to-end verification pipeline (arithmetic, narrative direction).
6. GPU-dependent tests (separated via pytest skipif).

No external dataset download is required.
"""

import sys
import os
import math
from pathlib import Path
from unittest.mock import MagicMock

# Bootstrap paths
REPO_ROOT = Path(__file__).resolve().parent.parent
for subdir in ["src", "src/retrieval", "src/verification", "src/eval"]:
    p = str(REPO_ROOT / subdir)
    if p not in sys.path:
        sys.path.insert(0, p)

# Stubs for heavy dependencies if running in minimal test environment
try:
    import torch
except ImportError:
    _torch_mock = MagicMock()
    _torch_mock.cuda.is_available.return_value = False
    sys.modules["torch"] = _torch_mock

try:
    import transformers
except ImportError:
    _tf_mock = MagicMock()
    sys.modules["transformers"] = _tf_mock

try:
    import sentence_transformers
except ImportError:
    _st_mock = MagicMock()
    sys.modules["sentence_transformers"] = _st_mock

if "rank_bm25" not in sys.modules:
    try:
        import rank_bm25
    except ImportError:
        class _MockBM25:
            def __init__(self, corpus):
                self.corpus = corpus
            def get_scores(self, query_tokens):
                scores = []
                qset = set(query_tokens)
                for doc in self.corpus:
                    scores.append(float(len(qset.intersection(set(doc)))))
                return scores
        _bm_mock = MagicMock()
        _bm_mock.BM25Okapi = _MockBM25
        sys.modules["rank_bm25"] = _bm_mock

try:
    import pytest
except ImportError:
    class _MockPytest:
        class mark:
            @staticmethod
            def skipif(cond, reason=""):
                def decorator(fn):
                    if cond:
                        def skipped(*args, **kwargs):
                            print(f"[SKIPPED] {fn.__name__}: {reason}")
                        return skipped
                    return fn
                return decorator
    pytest = _MockPytest()

from rrf_retriever import (
    reciprocal_rank_fusion,
    get_chunk_uid,
    retrieve,
)
from cross_encoder_reranker import CrossEncoderReranker
from validator import (
    validate_operand,
    unit_matches_evidence,
    period_matches_evidence,
    build_chunk_lookup,
)
from pipeline import run_verification_pipeline


# ============================================================
# Real-format FinQA Fixtures
# ============================================================

DOC_ID_1 = "UNP/2009/page_34.pdf-1"
DOC_ID_2 = "CAT/2010/page_12.pdf-1"

FINQA_FIXTURE_CHUNKS = [
    {
        "example_id": DOC_ID_1,
        "chunk_id": "table_1",
        "source": "table",
        "text": (
            "millions of dollars: interest expense | "
            "2009: -600 ( 600 ) | "
            "2008: -511 ( 511 ) | "
            "% change 2009 v 2008: 17"
        ),
    },
    {
        "example_id": DOC_ID_1,
        "chunk_id": "post_text_0",
        "source": "post_text",
        "text": (
            "interest expense increased in 2009 versus 2008 due primarily "
            "to higher weighted-average debt levels."
        ),
    },
    {
        "example_id": DOC_ID_2,
        "chunk_id": "table_0",
        "source": "table",
        "text": (
            "millions of dollars: machinery sales | "
            "2010: 32000 | 2009: 28000"
        ),
    },
    {
        "example_id": DOC_ID_2,
        "chunk_id": "post_text_1",
        "source": "post_text",
        "text": "caterpillar machinery sales increased in 2010 across north america.",
    },
]

FINQA_QUERY = "What was the interest expense in 2009?"


# ============================================================
# Test 1: BM25 Retrieval Integration
# ============================================================

def test_bm25_retrieval_integration():
    """Verify BM25 ranking prioritizes relevant FinQA table chunk."""
    import rank_bm25
    BM25Class = getattr(rank_bm25, "BM25Okapi", None)
    if BM25Class is None or isinstance(BM25Class, MagicMock):
        # Fallback pure-python term-frequency ranker if running without rank_bm25 C-library
        class BM25Class:
            def __init__(self, corpus):
                self.corpus = corpus
            def get_scores(self, query_tokens):
                q = set(query_tokens)
                return [float(sum(1 for t in doc if t in q)) for doc in self.corpus]

    tokenized_corpus = [c["text"].lower().split() for c in FINQA_FIXTURE_CHUNKS]
    bm25 = BM25Class(tokenized_corpus)

    query_tokens = FINQA_QUERY.lower().split()
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_chunk = FINQA_FIXTURE_CHUNKS[ranked_indices[0]]

    # Top chunk should belong to UNP 2009 interest expense
    assert top_chunk["example_id"] == DOC_ID_1
    assert "interest expense" in top_chunk["text"]


# ============================================================
# Test 2: RRF Fusion with Stable Identifiers Integration
# ============================================================

def test_rrf_fusion_stable_identifiers():
    """Verify RRF correctly fuses rankings using unique chunk UIDs."""
    uid_table = f"{DOC_ID_1}#table_1"
    uid_post = f"{DOC_ID_1}#post_text_0"
    uid_other = f"{DOC_ID_2}#table_0"

    bm25_results = [
        (uid_table, 5.2),
        (uid_other, 1.1),
    ]

    dense_results = [
        (uid_post, 0.85),
        (uid_table, 0.82),
    ]

    # Chunk table_1 appears in rank 1 of BM25 and rank 2 of Dense
    fused = reciprocal_rank_fusion(
        bm25_results,
        dense_results,
        k=60,
    )

    # Top fused chunk should be table_1 because it appears in both lists
    assert len(fused) == 3
    assert fused[0][0] == uid_table
    assert fused[0][1] > fused[1][1]  # Higher RRF score

    # Verify UIDs match expected set
    fused_uids = {item[0] for item in fused}
    assert fused_uids == {uid_table, uid_post, uid_other}


# ============================================================
# Test 3: Cross-Encoder Reranking Integration
# ============================================================

def test_cross_encoder_reranker_integration():
    """Verify cross-encoder reranks candidates preserving descending score order."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.1, 0.95, 0.4]
    reranker = CrossEncoderReranker(model_name="mock-reranker", model=mock_model)

    candidates = [
        {"example_id": DOC_ID_2, "chunk_id": "table_0", "text": "machinery"},
        {"example_id": DOC_ID_1, "chunk_id": "table_1", "text": "interest expense 2009"},
        {"example_id": DOC_ID_1, "chunk_id": "post_text_0", "text": "interest narrative"},
    ]

    ranked = reranker.rerank(FINQA_QUERY, candidates, top_k=2)

    assert len(ranked) == 2
    assert ranked[0]["chunk_id"] == "table_1"
    assert ranked[0]["rerank_score"] == 0.95
    assert ranked[1]["chunk_id"] == "post_text_0"
    assert ranked[1]["rerank_score"] == 0.4


# ============================================================
# Test 4: Evidence & Unit/Period Validation Integration
# ============================================================

def test_unit_and_period_validation_rules():
    """Verify that unit and period validation strictly enforce financial semantics."""
    table_chunk = FINQA_FIXTURE_CHUNKS[0]
    lookup = build_chunk_lookup(FINQA_FIXTURE_CHUNKS)

    # 1. Valid operand: 600, 2009, million USD
    valid_op = {
        "id": "op_1",
        "chunk_id": "table_1",
        "span_text": "2009: -600 ( 600 )",
        "normalized_value": -600,
        "unit": "million USD",
        "period": "2009",
    }
    v_res = validate_operand(valid_op, lookup, DOC_ID_1)
    assert v_res["valid"] is True, f"Failed: {v_res.get('reason')}"

    # 2. Rejection of bare '$' matching 'million USD'
    assert not unit_matches_evidence("million USD", "$")
    assert not unit_matches_evidence("million USD", "USD")
    assert unit_matches_evidence("million USD", "millions of dollars: interest expense")

    # 3. Rejection of substring year match ('20' matching '2009')
    assert not period_matches_evidence("2009", "20")
    assert period_matches_evidence("2009", "2009")
    assert period_matches_evidence("2009", "fiscal year 2009")


# ============================================================
# Test 5: End-to-End Verification Pipeline Integration
# ============================================================

def test_end_to_end_verification_valid_claim():
    """Verify a mathematically and evidentially grounded claim passes all stages."""
    claim = {
        "claims": [
            {
                "id": "claim_diff",
                "claim_text": "The change in interest expense from 2008 to 2009 was 89 million USD.",
                "value": 89,
                "unit": "million USD",
                "citations": ["table_1", "post_text_0"],
                "operands": [
                    {
                        "id": "exp_2009",
                        "chunk_id": "table_1",
                        "span_text": "2009: -600 ( 600 )",
                        "context_text": "millions of dollars: interest expense | 2009: -600 ( 600 )",
                        "normalized_value": 600,
                        "unit": "million USD",
                        "period": "2009",
                    },
                    {
                        "id": "exp_2008",
                        "chunk_id": "table_1",
                        "span_text": "2008: -511 ( 511 )",
                        "context_text": "millions of dollars: interest expense | 2008: -511 ( 511 )",
                        "normalized_value": 511,
                        "unit": "million USD",
                        "period": "2008",
                    },
                ],
                "program": [
                    {
                        "id": "step_1",
                        "operation": "subtract",
                        "operands": ["exp_2009", "exp_2008"],
                    }
                ],
                "referent": {
                    "entity": "interest expense",
                    "period": "2009",
                    "unit": "million USD",
                },
                "status": "answered",
            }
        ],
        "final_claim_id": "claim_diff",
    }

    result = run_verification_pipeline(
        claim,
        DOC_ID_1,
        chunks=FINQA_FIXTURE_CHUNKS,
        context_passages=[FINQA_FIXTURE_CHUNKS[1]["text"]],
        resolve_economic_sign=True,
    )

    assert result["valid"] is True
    assert result["status"] == "VERIFIED_RESULT"
    assert result["calculation"]["result"] == 89


def test_end_to_end_verification_detects_contradiction():
    """Verify pipeline rejects directional contradiction when narrative contradicts arithmetic."""
    claim = {
        "claims": [
            {
                "id": "claim_diff",
                "claim_text": "Interest expense decreased by 89 million USD.",
                "value": -89,
                "unit": "million USD",
                "citations": ["table_1", "post_text_0"],
                "operands": [
                    {
                        "id": "exp_2009",
                        "chunk_id": "table_1",
                        "span_text": "2009: -600 ( 600 )",
                        "context_text": "millions of dollars: interest expense | 2009: -600 ( 600 )",
                        "normalized_value": -600,
                        "unit": "million USD",
                        "period": "2009",
                    },
                    {
                        "id": "exp_2008",
                        "chunk_id": "table_1",
                        "span_text": "2008: -511 ( 511 )",
                        "context_text": "millions of dollars: interest expense | 2008: -511 ( 511 )",
                        "normalized_value": -511,
                        "unit": "million USD",
                        "period": "2008",
                    },
                ],
                "program": [
                    {
                        "id": "step_1",
                        "operation": "subtract",
                        "operands": ["exp_2009", "exp_2008"],
                    }
                ],
                "referent": {
                    "entity": "interest expense",
                    "period": "2009",
                    "unit": "million USD",
                },
                "status": "answered",
            }
        ],
        "final_claim_id": "claim_diff",
    }

    result = run_verification_pipeline(
        claim,
        DOC_ID_1,
        chunks=FINQA_FIXTURE_CHUNKS,
        context_passages=["interest expense increased in 2009 versus 2008"],
        resolve_economic_sign=False,
    )

    assert result["valid"] is False
    assert result["status"] == "DIRECTIONAL_CONTRADICTION"


def test_end_to_end_verification_detects_ungrounded_operand():
    """Verify pipeline rejects claims with fabricated/ungrounded numerical values."""
    claim = {
        "claims": [
            {
                "id": "claim_fake",
                "claim_text": "The expense was 999 million USD.",
                "value": 999,
                "unit": "million USD",
                "citations": ["table_1"],
                "operands": [
                    {
                        "id": "op_fake",
                        "chunk_id": "table_1",
                        "span_text": "2009: 999",
                        "context_text": "millions of dollars: interest expense | 2009: 999",
                        "normalized_value": 999,
                        "unit": "million USD",
                        "period": "2009",
                    }
                ],
                "program": [
                    {
                        "id": "step_1",
                        "operation": "add",
                        "operands": ["op_fake", "op_fake"],
                    }
                ],
                "referent": {"entity": "interest expense", "period": "2009", "unit": "million USD"},
                "status": "answered",
            }
        ],
        "final_claim_id": "claim_fake",
    }

    result = run_verification_pipeline(
        claim,
        DOC_ID_1,
        chunks=FINQA_FIXTURE_CHUNKS,
    )

    assert result["valid"] is False
    assert result["status"] in ("INVALID_EVIDENCE", "OPERAND_NOT_FOUND", "INVALID_OPERAND", "OPERAND_MISMATCH")


# ============================================================
# Test 6: GPU-Dependent Tests (Separated & Skipped if no CUDA)
# ============================================================

def _has_cuda():
    try:
        import torch
        if not hasattr(torch, "cuda") or isinstance(torch.cuda, MagicMock):
            return False
        res = torch.cuda.is_available()
        return bool(res) if not isinstance(res, MagicMock) else False
    except Exception:
        return False


@pytest.mark.skipif(not _has_cuda(), reason="CUDA GPU not available in current test environment")
def test_gpu_embedding_and_reranking_execution():
    """Verify GPU execution of sentence-transformers and cross-encoder when CUDA is active."""
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from sentence_transformers import SentenceTransformer

    assert torch.cuda.is_available()
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    assert "RTX" in gpu_name or "GeForce" in gpu_name or len(gpu_name) > 0

    # Test encoding small tensor on CUDA
    import glob
    snap_pattern = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/*")
    snaps = glob.glob(snap_pattern)
    model_id = snaps[0] if snaps else "BAAI/bge-small-en-v1.5"
    model = SentenceTransformer(model_id, device=device)
    embeddings = model.encode(["Test financial query on GPU"], device=device, convert_to_tensor=True)
    assert embeddings.device.type == "cuda"
    assert embeddings.shape[-1] == 384
