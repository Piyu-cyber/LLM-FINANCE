"""test_retrieval_eval.py — Unit tests for Phase 1 dual-level recall metrics.

Run with:
    $env:PYTHONPATH="src;src/retrieval;src/verification;src/eval"
    python -m pytest src/eval/test_retrieval_eval.py -v

All tests use synthetic fixtures; no FinQA corpus file is required.
"""

import sys
import os
import math
from unittest.mock import MagicMock

# Ensure the eval module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

# Stub heavy dependencies so the test file can be collected without
# the full ML stack installed in the system Python environment.
# NOTE: we intentionally do NOT mock numpy here to avoid breaking
# pytest.approx (which calls isinstance(val, np.bool_)).
# Instead, retrieval_eval.py is imported after inserting a minimal stub.
for _mod in ("rank_bm25", "sentence_transformers", "torch"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# numpy stub: only stub if truly absent; provide a real bool_ to keep pytest happy
if "numpy" not in sys.modules:
    _np_stub = MagicMock()
    _np_stub.bool_ = bool   # make isinstance(x, np.bool_) work
    sys.modules["numpy"] = _np_stub

import pytest

from retrieval_eval import (
    create_metrics,
    update_metrics,
    page_hit,
    get_example_id,
    get_gold_evidence,
    evidence_matches,
    metrics_to_dict,
)


# ============================================================
# Fixtures
# ============================================================

EXAMPLE_ID = "UNP/2009/page_34.pdf-1"

GOLD_EXAMPLE = {
    "id": EXAMPLE_ID,
    "qa": {
        "question": "What is the change in interest expense from 2008 to 2009?",
        "gold_inds": {
            "table_1": "millions of dollars: interest expense | 2009: -600 ( 600 ) | 2008: -511 ( 511 )",
            "post_text_4": "interest expense 2013 interest expense increased in 2009 versus 2008",
        },
    },
}

MATCHING_TABLE_CHUNK = {
    "example_id": EXAMPLE_ID,
    "chunk_id": "table_1",
    "source": "table",
    "text": "millions of dollars: interest expense | 2009: -600 ( 600 ) | 2008: -511 ( 511 )",
}

MATCHING_TEXT_CHUNK = {
    "example_id": EXAMPLE_ID,
    "chunk_id": "post_text_4",
    "source": "post_text",
    "text": "interest expense 2013 interest expense increased in 2009 versus 2008",
}

WRONG_COMPANY_CHUNK = {
    "example_id": "GS/2009/page_5.pdf-1",
    "chunk_id": "table_0",
    "source": "table",
    "text": "interest expense | 2009: -300 | 2008: -250",
}

WRONG_YEAR_CHUNK = {
    "example_id": "UNP/2011/page_10.pdf-1",
    "chunk_id": "table_0",
    "source": "table",
    "text": "interest expense | 2011: -700 | 2010: -620",
}

UNRELATED_CHUNK = {
    "example_id": "JPM/2015/page_2.pdf-1",
    "chunk_id": "text_0",
    "source": "pre_text",
    "text": "jpmorgan chase reports record earnings for fiscal year 2015.",
}


# ============================================================
# get_example_id
# ============================================================

class TestGetExampleId:
    def test_returns_id_string(self):
        assert get_example_id(GOLD_EXAMPLE) == EXAMPLE_ID

    def test_missing_id_returns_empty(self):
        assert get_example_id({}) == ""

    def test_non_string_id_is_coerced(self):
        example = {"id": 42}
        assert get_example_id(example) == "42"


# ============================================================
# get_gold_evidence
# ============================================================

class TestGetGoldEvidence:
    def test_dict_gold_inds(self):
        ev = get_gold_evidence(GOLD_EXAMPLE)
        assert len(ev) == 2
        assert any("interest expense" in e for e in ev)

    def test_list_gold_inds(self):
        example = {"qa": {"gold_inds": ["some text", "other text"]}}
        ev = get_gold_evidence(example)
        assert ev == ["some text", "other text"]

    def test_string_gold_inds(self):
        example = {"qa": {"gold_inds": "just a string"}}
        ev = get_gold_evidence(example)
        assert ev == ["just a string"]

    def test_no_qa_returns_empty(self):
        assert get_gold_evidence({}) == []

    def test_empty_gold_inds_returns_empty(self):
        example = {"qa": {"gold_inds": {}}}
        assert get_gold_evidence(example) == []


# ============================================================
# evidence_matches
# ============================================================

class TestEvidenceMatches:
    def test_exact_match(self):
        gold = ["interest expense increased in 2009"]
        assert evidence_matches("interest expense increased in 2009", gold)

    def test_gold_substring_of_retrieved(self):
        # gold text is contained in a longer retrieved chunk
        gold = ["interest expense increased"]
        retrieved = "interest expense increased in 2009 versus 2008 due to higher debt"
        assert evidence_matches(retrieved, gold)

    def test_retrieved_substring_of_gold(self):
        # shorter retrieved chunk is contained in longer gold text
        gold = ["interest expense increased in 2009 versus 2008 due to higher debt"]
        retrieved = "interest expense increased in 2009"
        assert evidence_matches(retrieved, gold)

    def test_case_insensitive(self):
        gold = ["INTEREST EXPENSE"]
        assert evidence_matches("interest expense increased", gold)

    def test_no_match(self):
        gold = ["revenue from operations"]
        assert not evidence_matches("interest expense increased", gold)

    def test_empty_gold_no_match(self):
        assert not evidence_matches("some text", [])

    def test_non_string_gold_entries_skipped(self):
        gold = [None, 42, "interest expense"]
        assert evidence_matches("interest expense in 2009", gold)


# ============================================================
# page_hit
# ============================================================

class TestPageHit:
    def test_same_document_is_hit(self):
        assert page_hit(MATCHING_TABLE_CHUNK, EXAMPLE_ID)

    def test_wrong_company_is_miss(self):
        assert not page_hit(WRONG_COMPANY_CHUNK, EXAMPLE_ID)

    def test_wrong_year_is_miss(self):
        assert not page_hit(WRONG_YEAR_CHUNK, EXAMPLE_ID)

    def test_different_page_same_company_year(self):
        # Different page number should still be a miss — page_hit is strict
        chunk = {
            "example_id": "UNP/2009/page_35.pdf-1",
            "chunk_id": "table_0",
            "source": "table",
            "text": "some other row",
        }
        assert not page_hit(chunk, EXAMPLE_ID)

    def test_empty_gold_id_never_hits(self):
        assert not page_hit(MATCHING_TABLE_CHUNK, "")

    def test_chunk_without_example_id(self):
        chunk = {"text": "something"}
        assert not page_hit(chunk, EXAMPLE_ID)


# ============================================================
# create_metrics / update_metrics
# ============================================================

class TestMetrics:
    def test_create_metrics_has_all_keys(self):
        m = create_metrics()
        assert "evaluated" in m
        assert "no_gold" in m
        assert "span_recall_at_1" in m
        assert "span_recall_at_5" in m
        assert "span_recall_at_10" in m
        assert "span_mrr" in m
        assert "page_recall_at_1" in m
        assert "page_recall_at_5" in m
        assert "page_recall_at_10" in m
        assert "page_mrr" in m

    def test_update_both_found_at_rank_1(self):
        m = create_metrics()
        update_metrics(1, 1, m)

        assert m["evaluated"] == 1
        assert m["span_recall_at_1"] == 1
        assert m["span_recall_at_5"] == 1
        assert m["span_recall_at_10"] == 1
        assert math.isclose(m["span_mrr"], 1.0)
        assert m["page_recall_at_1"] == 1
        assert m["page_recall_at_5"] == 1
        assert m["page_recall_at_10"] == 1
        assert math.isclose(m["page_mrr"], 1.0)

    def test_update_span_at_7_page_at_2(self):
        m = create_metrics()
        update_metrics(7, 2, m)

        # Span
        assert m["span_recall_at_1"] == 0
        assert m["span_recall_at_5"] == 0
        assert m["span_recall_at_10"] == 1
        assert math.isclose(m["span_mrr"], 1 / 7)

        # Page
        assert m["page_recall_at_1"] == 0
        assert m["page_recall_at_5"] == 1
        assert m["page_recall_at_10"] == 1
        assert math.isclose(m["page_mrr"], 1 / 2)

    def test_update_span_not_found_page_found(self):
        m = create_metrics()
        update_metrics(None, 3, m)

        assert m["span_recall_at_10"] == 0
        assert m["span_mrr"] == 0.0
        assert m["page_recall_at_5"] == 1
        assert math.isclose(m["page_mrr"], 1 / 3)

    def test_update_neither_found(self):
        m = create_metrics()
        update_metrics(None, None, m)

        assert m["evaluated"] == 1
        assert m["span_recall_at_10"] == 0
        assert m["page_recall_at_10"] == 0

    def test_accumulation_over_multiple_examples(self):
        m = create_metrics()
        update_metrics(1, 1, m)   # perfect hit: span@1, page@1
        update_metrics(11, 1, m)  # page hit@1, span rank 11 (outside top-10)
        update_metrics(None, None, m)  # both missed

        assert m["evaluated"] == 3
        assert m["span_recall_at_10"] == 1
        assert m["page_recall_at_10"] == 2
        # span MRR: 1/1 + 1/11 = 1.0909...
        assert math.isclose(m["span_mrr"], 1.0 + 1.0 / 11)
        # page MRR: 1/1 + 1/1 = 2.0
        assert math.isclose(m["page_mrr"], 2.0)


# ============================================================
# metrics_to_dict
# ============================================================

class TestMetricsToDict:
    def test_zero_evaluated_returns_minimal_dict(self):
        m = create_metrics()
        d = metrics_to_dict("BM25", m)
        assert d["method"] == "BM25"
        assert d["evaluated"] == 0
        assert "span_recall_at_10" not in d

    def test_populated_metrics_serialise_correctly(self):
        m = create_metrics()
        update_metrics(1, 1, m)
        update_metrics(5, 3, m)

        d = metrics_to_dict("RRF", m)
        assert d["method"] == "RRF"
        assert d["evaluated"] == 2
        assert math.isclose(d["span_recall_at_1"], 0.5)
        assert math.isclose(d["span_recall_at_5"], 1.0)
        assert math.isclose(d["page_recall_at_5"], 1.0)
        assert "span_mrr" in d
        assert "page_mrr" in d
