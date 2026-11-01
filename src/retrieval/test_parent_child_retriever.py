"""test_parent_child_retriever.py — Tests for Phase 1 parent-child expansion.

Run with:
    $env:PYTHONPATH="src;src/retrieval;src/verification;src/eval"
    python -m pytest src/retrieval/test_parent_child_retriever.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from parent_child_retriever import (
    build_corpus_index,
    expand_chunk,
    expand_chunks,
    build_llm_context,
)


# ============================================================
# Fixtures
# ============================================================

EXAMPLE_ID = "UNP/2009/page_34.pdf-1"
OTHER_ID = "GS/2015/page_5.pdf-1"

CORPUS = [
    # UNP 2009 pre_text
    {
        "example_id": EXAMPLE_ID, "chunk_id": "text_0", "source": "pre_text",
        "parent_id": EXAMPLE_ID, "position": 0,
        "text": "the following table shows interest expense in millions of dollars .",
    },
    {
        "example_id": EXAMPLE_ID, "chunk_id": "text_1", "source": "pre_text",
        "parent_id": EXAMPLE_ID, "position": 1,
        "text": "union pacific railroad company financial summary .",
    },
    # UNP 2009 table
    {
        "example_id": EXAMPLE_ID, "chunk_id": "table_0", "source": "table",
        "parent_id": EXAMPLE_ID, "position": 0,
        "text": "[DOC: UNP 2009] interest expense | 2009: -600 | 2008: -511",
    },
    {
        "example_id": EXAMPLE_ID, "chunk_id": "table_1", "source": "table",
        "parent_id": EXAMPLE_ID, "position": 1,
        "text": "[DOC: UNP 2009] other expenses | 2009: -100 | 2008: -90",
    },
    # UNP 2009 post_text
    {
        "example_id": EXAMPLE_ID, "chunk_id": "post_text_0", "source": "post_text",
        "parent_id": EXAMPLE_ID, "position": 0,
        "text": "interest expense increased in 2009 versus 2008 .",
    },
    # GS 2015 (different example)
    {
        "example_id": OTHER_ID, "chunk_id": "text_0", "source": "pre_text",
        "parent_id": OTHER_ID, "position": 0,
        "text": "goldman sachs reports record revenues .",
    },
]

CHILD_TABLE_CHUNK = CORPUS[2]  # table_0 from UNP 2009


# ============================================================
# build_corpus_index
# ============================================================

class TestBuildCorpusIndex:
    def test_groups_by_example_id(self):
        index = build_corpus_index(CORPUS)
        assert EXAMPLE_ID in index
        assert OTHER_ID in index

    def test_correct_chunk_count(self):
        index = build_corpus_index(CORPUS)
        assert len(index[EXAMPLE_ID]) == 5
        assert len(index[OTHER_ID]) == 1

    def test_chunks_preserve_order(self):
        index = build_corpus_index(CORPUS)
        sources = [c["source"] for c in index[EXAMPLE_ID]]
        # pre_text comes first in CORPUS fixture
        assert sources[0] == "pre_text"

    def test_empty_corpus(self):
        assert build_corpus_index([]) == {}


# ============================================================
# expand_chunk
# ============================================================

class TestExpandChunk:
    def setup_method(self):
        self.index = build_corpus_index(CORPUS)

    def test_returns_expected_keys(self):
        result = expand_chunk(CHILD_TABLE_CHUNK, self.index)
        assert "child_chunk" in result
        assert "parent_text" in result
        assert "table_chunks" in result
        assert "text_chunks" in result
        assert "example_id" in result

    def test_child_chunk_preserved(self):
        result = expand_chunk(CHILD_TABLE_CHUNK, self.index)
        assert result["child_chunk"] is CHILD_TABLE_CHUNK

    def test_example_id_matches(self):
        result = expand_chunk(CHILD_TABLE_CHUNK, self.index)
        assert result["example_id"] == EXAMPLE_ID

    def test_table_chunks_included(self):
        result = expand_chunk(CHILD_TABLE_CHUNK, self.index)
        assert len(result["table_chunks"]) == 2

    def test_text_chunks_included(self):
        result = expand_chunk(CHILD_TABLE_CHUNK, self.index)
        # 2 pre_text + 1 post_text = 3, capped by max_text_sentences=5
        assert len(result["text_chunks"]) == 3

    def test_max_text_sentences_respected(self):
        result = expand_chunk(
            CHILD_TABLE_CHUNK,
            self.index,
            max_text_sentences=1
        )
        assert len(result["text_chunks"]) == 1

    def test_parent_text_non_empty(self):
        result = expand_chunk(CHILD_TABLE_CHUNK, self.index)
        assert len(result["parent_text"]) > 0

    def test_parent_text_contains_table_data(self):
        result = expand_chunk(CHILD_TABLE_CHUNK, self.index)
        assert "interest expense" in result["parent_text"]

    def test_exclude_table(self):
        result = expand_chunk(
            CHILD_TABLE_CHUNK,
            self.index,
            include_table=False
        )
        assert result["table_chunks"] == []
        assert "interest expense" not in result["parent_text"] or \
               "text_chunks" in result

    def test_unknown_example_id_returns_empty(self):
        fake_chunk = {"example_id": "FAKE/2099/page_1.pdf-1", "parent_id": "FAKE/2099/page_1.pdf-1", "text": ""}
        result = expand_chunk(fake_chunk, self.index)
        assert result["parent_text"] == ""
        assert result["table_chunks"] == []
        assert result["text_chunks"] == []


# ============================================================
# expand_chunks (batch)
# ============================================================

class TestExpandChunks:
    def setup_method(self):
        self.index = build_corpus_index(CORPUS)

    def test_single_chunk(self):
        results = expand_chunks([CHILD_TABLE_CHUNK], self.index)
        assert len(results) == 1

    def test_deduplication_by_parent(self):
        # Two chunks from the same parent should yield one expanded result
        table_chunks = [CORPUS[2], CORPUS[3]]  # table_0 and table_1 of UNP/2009
        results = expand_chunks(table_chunks, self.index, deduplicate_by_parent=True)
        assert len(results) == 1

    def test_no_deduplication(self):
        table_chunks = [CORPUS[2], CORPUS[3]]
        results = expand_chunks(table_chunks, self.index, deduplicate_by_parent=False)
        assert len(results) == 2

    def test_multi_parent(self):
        # One chunk from each example
        chunks = [CORPUS[2], CORPUS[5]]  # UNP table_0, GS text_0
        results = expand_chunks(chunks, self.index)
        parent_ids = [r["example_id"] for r in results]
        assert EXAMPLE_ID in parent_ids
        assert OTHER_ID in parent_ids

    def test_empty_input(self):
        assert expand_chunks([], self.index) == []


# ============================================================
# build_llm_context
# ============================================================

class TestBuildLlmContext:
    def setup_method(self):
        index = build_corpus_index(CORPUS)
        self.expanded = expand_chunks([CHILD_TABLE_CHUNK], index)

    def test_returns_string(self):
        ctx = build_llm_context(self.expanded)
        assert isinstance(ctx, str)

    def test_contains_evidence_header(self):
        ctx = build_llm_context(self.expanded)
        assert "[Evidence 1]" in ctx

    def test_contains_example_id(self):
        ctx = build_llm_context(self.expanded)
        assert EXAMPLE_ID in ctx

    def test_max_results_respected(self):
        # Duplicate expanded results to have multiple
        doubled = self.expanded * 3
        ctx = build_llm_context(doubled, max_results=2)
        assert "[Evidence 1]" in ctx
        assert "[Evidence 2]" in ctx
        assert "[Evidence 3]" not in ctx

    def test_empty_input(self):
        assert build_llm_context([]) == ""
