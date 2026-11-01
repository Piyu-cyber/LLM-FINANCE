"""test_chunker.py — Unit tests for Phase 1 enhanced chunker.

Run with:
    $env:PYTHONPATH="src;src/retrieval;src/verification;src/eval"
    python -m pytest src/retrieval/test_chunker.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from chunker import (
    _parse_company_year,
    _infer_units,
    create_text_chunks,
    create_table_chunks,
    create_chunks,
)


# ============================================================
# Fixtures
# ============================================================

EXAMPLE_ID = "UNP/2009/page_34.pdf-1"

SAMPLE_EXAMPLE = {
    "id": EXAMPLE_ID,
    "pre_text": [
        "the following table shows interest expense in millions of dollars .",
        "union pacific railroad company financial summary .",
    ],
    "post_text": [
        "interest expense increased in 2009 versus 2008 due primarily to higher weighted-average debt levels .",
    ],
    "table": [
        ["", "2009", "2008", "2007"],
        ["interest expense", "-600 ( 600 )", "-511 ( 511 )", "-482 ( 482 )"],
        ["other expenses", "-100 ( 100 )", "-90 ( 90 )", "-80 ( 80 )"],
    ],
    "qa": {
        "question": "What is the change in interest expense from 2008 to 2009?",
        "gold_inds": {
            "table_0": "interest expense | 2009: -600 ( 600 ) | 2008: -511 ( 511 )",
        },
    },
}

EXAMPLE_NO_UNIT = {
    "id": "GS/2015/page_5.pdf-1",
    "pre_text": ["Goldman Sachs reports record revenues."],
    "post_text": [],
    "table": [
        ["", "2015", "2014"],
        ["revenue", "30000", "28000"],
    ],
    "qa": {"question": "Revenue?", "gold_inds": {}},
}


# ============================================================
# _parse_company_year
# ============================================================

class TestParseCompanyYear:
    def test_standard_id(self):
        ticker, year = _parse_company_year("UNP/2009/page_34.pdf-1")
        assert ticker == "UNP"
        assert year == "2009"

    def test_lowercase_ticker_is_uppercased(self):
        ticker, year = _parse_company_year("unp/2009/page_1.pdf-1")
        assert ticker == "UNP"

    def test_malformed_id_returns_empty(self):
        ticker, year = _parse_company_year("")
        assert ticker == ""
        assert year == ""

    def test_only_ticker(self):
        ticker, year = _parse_company_year("UNP")
        assert ticker == "UNP"
        assert year == ""


# ============================================================
# _infer_units
# ============================================================

class TestInferUnits:
    def test_detects_millions_of_dollars(self):
        pre_text = ["results in millions of dollars", "other text"]
        units = _infer_units(pre_text)
        assert "million" in units.lower()

    def test_detects_billions(self):
        pre_text = ["revenues in billions of dollars"]
        units = _infer_units(pre_text)
        assert "billion" in units.lower()

    def test_no_unit_returns_empty(self):
        pre_text = ["some random text with no unit mention"]
        units = _infer_units(pre_text)
        assert units == ""

    def test_empty_pre_text(self):
        assert _infer_units([]) == ""


# ============================================================
# create_text_chunks
# ============================================================

class TestCreateTextChunks:
    def test_returns_list(self):
        chunks = create_text_chunks(SAMPLE_EXAMPLE)
        assert isinstance(chunks, list)

    def test_pre_and_post_text_included(self):
        chunks = create_text_chunks(SAMPLE_EXAMPLE)
        sources = {c["source"] for c in chunks}
        assert "pre_text" in sources
        assert "post_text" in sources

    def test_chunk_has_required_keys(self):
        chunks = create_text_chunks(SAMPLE_EXAMPLE)
        required = {"example_id", "chunk_id", "source", "position", "text",
                    "company", "year", "parent_id"}
        for chunk in chunks:
            assert required.issubset(chunk.keys()), f"Missing keys in {chunk}"

    def test_metadata_fields_populated(self):
        chunks = create_text_chunks(SAMPLE_EXAMPLE)
        for chunk in chunks:
            assert chunk["company"] == "UNP"
            assert chunk["year"] == "2009"
            assert chunk["example_id"] == EXAMPLE_ID
            assert chunk["parent_id"] == EXAMPLE_ID

    def test_empty_text_skipped(self):
        example = {
            "id": EXAMPLE_ID,
            "pre_text": ["", "   ", "valid text"],
            "post_text": [],
        }
        chunks = create_text_chunks(example)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "valid text"


# ============================================================
# create_table_chunks
# ============================================================

class TestCreateTableChunks:
    def test_returns_one_chunk_per_data_row(self):
        # Table has 1 header + 2 data rows
        chunks = create_table_chunks(SAMPLE_EXAMPLE)
        assert len(chunks) == 2

    def test_chunk_has_required_keys(self):
        chunks = create_table_chunks(SAMPLE_EXAMPLE)
        required = {"example_id", "chunk_id", "source", "position", "text",
                    "company", "year", "parent_id", "units"}
        for chunk in chunks:
            assert required.issubset(chunk.keys()), f"Missing keys in {chunk}"

    def test_metadata_correct(self):
        chunks = create_table_chunks(SAMPLE_EXAMPLE)
        for chunk in chunks:
            assert chunk["company"] == "UNP"
            assert chunk["year"] == "2009"
            assert chunk["example_id"] == EXAMPLE_ID
            assert chunk["parent_id"] == EXAMPLE_ID

    def test_doc_prefix_in_text(self):
        chunks = create_table_chunks(SAMPLE_EXAMPLE)
        for chunk in chunks:
            assert "[DOC: UNP 2009]" in chunk["text"]

    def test_unit_prefix_in_text_when_detected(self):
        chunks = create_table_chunks(SAMPLE_EXAMPLE)
        # pre_text says "in millions of dollars"
        for chunk in chunks:
            assert "[UNIT:" in chunk["text"]

    def test_no_unit_prefix_when_not_detected(self):
        chunks = create_table_chunks(EXAMPLE_NO_UNIT)
        for chunk in chunks:
            assert "[UNIT:" not in chunk["text"]

    def test_column_headers_included(self):
        chunks = create_table_chunks(SAMPLE_EXAMPLE)
        # First row is "interest expense" row
        first = chunks[0]["text"]
        assert "2009" in first
        assert "2008" in first

    def test_empty_table_returns_empty_list(self):
        example = {
            "id": EXAMPLE_ID,
            "pre_text": [],
            "post_text": [],
            "table": [],
            "qa": {},
        }
        assert create_table_chunks(example) == []

    def test_header_only_table_returns_empty_list(self):
        example = {
            "id": EXAMPLE_ID,
            "pre_text": [],
            "post_text": [],
            "table": [["col_a", "col_b"]],
            "qa": {},
        }
        assert create_table_chunks(example) == []


# ============================================================
# create_chunks (combined)
# ============================================================

class TestCreateChunks:
    def test_combines_text_and_table(self):
        chunks = create_chunks(SAMPLE_EXAMPLE)
        sources = {c["source"] for c in chunks}
        assert "pre_text" in sources
        assert "post_text" in sources
        assert "table" in sources

    def test_all_chunks_have_example_id(self):
        chunks = create_chunks(SAMPLE_EXAMPLE)
        for chunk in chunks:
            assert chunk["example_id"] == EXAMPLE_ID
