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
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

# numpy stub: only stub if truly absent; provide a real bool_ to keep pytest happy
if "numpy" not in sys.modules:
    try:
        import numpy
    except ImportError:
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


# ============================================================
# Phase 1: Robust Metric Definitions & Multi-Span Verification
# ============================================================

from retrieval_eval import (
    span_matches,
    normalize_evidence_text,
    compute_example_metrics,
)


class TestSpanMatchesRobustness:
    """Verifies that span matching rejects unsafe substrings and handles normalization."""

    def test_exact_and_contained_gold(self):
        gold = "interest expense increased to $600 million in 2009"
        chunk = "operating results: interest expense increased to $600 million in 2009 compared to $511 million in 2008."
        assert span_matches(chunk, gold)

    def test_unsafe_short_retrieved_substring_rejected(self):
        # A tiny retrieved chunk of 1-3 tokens that happens to appear in a long gold sentence
        # should NOT count as finding the gold evidence (prevents metric inflation).
        gold = "net income was $1,500 million for the year ended december 31, 2019."
        short_chunk_1 = "net income"
        short_chunk_2 = "million"
        short_chunk_3 = "the year ended"
        assert not span_matches(short_chunk_1, gold)
        assert not span_matches(short_chunk_2, gold)
        assert not span_matches(short_chunk_3, gold)

    def test_substantial_retrieved_substring_accepted(self):
        # A retrieved chunk with >= 4 tokens that is a substantial excerpt of gold is accepted
        gold = "net income attributable to common shareholders was $1,500 million for 2019"
        chunk = "net income attributable to common"
        assert span_matches(chunk, gold)

    def test_normalization_whitespace_and_case(self):
        gold = "Total   Revenue\n\t$12,500"
        chunk = "Item 8. total revenue $12,500 from continuing operations."
        assert span_matches(chunk, gold)

    def test_empty_or_none_inputs(self):
        assert not span_matches("", "some gold text")
        assert not span_matches("some chunk text", "")
        assert not span_matches(None, "some gold text")
        assert not span_matches("some chunk text", None)


class TestComputeExampleMetricsMultiSpan:
    """Tests distinguishing Hit@K from true multi-span Recall@K and independent page metrics."""

    def test_multi_span_recall_vs_hit_distinction(self):
        gold_spans = [
            "operating income of $500 million in 2020",
            "net income decreased by 10 percent",
        ]
        doc_id = "AAPL/2020/page_12"

        # Chunk 1 matches span 0
        chunk_1 = {
            "example_id": doc_id,
            "text": "we achieved operating income of $500 million in 2020 through efficiency.",
        }
        # Chunk 2 matches nothing
        chunk_2 = {
            "example_id": "MSFT/2020/page_1",
            "text": "unrelated software license revenue increased.",
        }
        # Chunk 3 matches span 1
        chunk_3 = {
            "example_id": doc_id,
            "text": "overall net income decreased by 10 percent due to higher costs.",
        }

        retrieved = [chunk_1, chunk_2, chunk_3]
        metrics = compute_example_metrics(retrieved, gold_spans, doc_id, top_ks=(1, 2, 3))

        # At K=1: 1 out of 2 gold spans found
        assert metrics["span_hit_at_1"] == 1.0
        assert math.isclose(metrics["span_recall_at_1"], 0.5)

        # At K=2: still only 1 out of 2 gold spans found (chunk 2 was irrelevant)
        assert metrics["span_hit_at_2"] == 1.0
        assert math.isclose(metrics["span_recall_at_2"], 0.5)

        # At K=3: both gold spans found
        assert metrics["span_hit_at_3"] == 1.0
        assert math.isclose(metrics["span_recall_at_3"], 1.0)

    def test_independent_page_and_span_relevance(self):
        gold_spans = ["long-term debt totaled $4,000 million"]
        gold_doc = "CAT/2018/page_55"

        # Case A: chunk from gold page, but irrelevant text
        chunk_page_only = {
            "example_id": gold_doc,
            "text": "forward looking statements disclaimer and risk factors.",
        }
        res_a = compute_example_metrics([chunk_page_only], gold_spans, gold_doc, top_ks=(1,))
        assert res_a["page_hit_at_1"] == 1.0
        assert res_a["span_hit_at_1"] == 0.0
        assert res_a["span_recall_at_1"] == 0.0

        # Case B: chunk has gold text, but from different doc/page (e.g. historical quote in another file)
        chunk_span_only = {
            "example_id": "DE/2018/page_10",
            "text": "long-term debt totaled $4,000 million across credit facilities.",
        }
        res_b = compute_example_metrics([chunk_span_only], gold_spans, gold_doc, top_ks=(1,))
        assert res_b["page_hit_at_1"] == 0.0
        assert res_b["span_hit_at_1"] == 1.0
        assert res_b["span_recall_at_1"] == 1.0

    def test_duplicate_matching_chunks_do_not_overcount_recall(self):
        gold_spans = ["revenue was $100 million", "expenses were $80 million"]
        doc_id = "DOC/1"

        # Two chunks both match span 0
        chunk_1 = {"example_id": doc_id, "text": "revenue was $100 million in 2021"}
        chunk_2 = {"example_id": doc_id, "text": "table row: revenue was $100 million"}

        res = compute_example_metrics([chunk_1, chunk_2], gold_spans, doc_id, top_ks=(2,))
        # Only 1 distinct gold span was covered
        assert math.isclose(res["span_recall_at_2"], 0.5)

    def test_empty_gold_spans_handled_safely(self):
        chunk = {"example_id": "DOC/1", "text": "some text"}
        res = compute_example_metrics([chunk], [], "DOC/1", top_ks=(1, 5))
        assert res["span_recall_at_1"] == 0.0
        assert res["span_hit_at_1"] == 0.0
        assert res["page_hit_at_1"] == 1.0
