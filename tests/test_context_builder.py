"""test_context_builder.py — Comprehensive tests for ContextBuilder."""

import pytest

try:
    from src.generation.context_builder import (
        ContextBuildResult,
        ContextBuilder,
        default_token_counter,
        format_child_chunk,
        format_parent_expanded_item,
    )
except ImportError:
    from context_builder import (
        ContextBuildResult,
        ContextBuilder,
        default_token_counter,
        format_child_chunk,
        format_parent_expanded_item,
    )


@pytest.fixture
def sample_child_chunks():
    return [
        {
            "example_id": "AAPL_2019",
            "chunk_id": "table_row_0",
            "source": "table",
            "company": "AAPL",
            "year": "2019",
            "text": "Total Net Sales | $ 260,174 | $ 265,595",
        },
        {
            "example_id": "AAPL_2019",
            "chunk_id": "text_0",
            "source": "pre_text",
            "company": "AAPL",
            "year": "2019",
            "text": "All monetary amounts are in millions of US dollars.",
        },
        {
            "example_id": "MSFT_2019",
            "chunk_id": "table_row_1",
            "source": "table",
            "company": "MSFT",
            "year": "2019",
            "text": "Operating Income | $ 42,959 | $ 35,058",
        },
    ]


@pytest.fixture
def sample_corpus_index(sample_child_chunks):
    return {
        "AAPL_2019": [
            sample_child_chunks[0],
            sample_child_chunks[1],
            {
                "example_id": "AAPL_2019",
                "chunk_id": "post_text_0",
                "source": "post_text",
                "company": "AAPL",
                "year": "2019",
                "text": "Gross margin was 37.8% in 2019.",
            },
        ],
        "MSFT_2019": [
            sample_child_chunks[2],
            {
                "example_id": "MSFT_2019",
                "chunk_id": "text_1",
                "source": "pre_text",
                "company": "MSFT",
                "year": "2019",
                "text": "Results reflect adoption of new revenue standards.",
            },
        ],
    }


def test_format_child_chunk():
    chunk = {
        "example_id": "GOOG_2020",
        "chunk_id": "table_row_5",
        "source": "table",
        "company": "GOOG",
        "year": "2020",
        "text": "Revenues | $ 182,527 | $ 161,857",
    }
    formatted = format_child_chunk(chunk)
    assert "[Evidence Chunk: table_row_5]" in formatted
    assert "Document: GOOG_2020" in formatted
    assert "Source: table" in formatted
    assert "Company: GOOG" in formatted
    assert "Year: 2020" in formatted
    assert "Revenues | $ 182,527 | $ 161,857" in formatted


def test_child_only_context_assembly(sample_child_chunks):
    builder = ContextBuilder(default_mode="child_only")
    res = builder.build_context(sample_child_chunks)

    assert isinstance(res, ContextBuildResult)
    assert res.mode == "child_only"
    assert len(res.chunks_included) == 3
    assert len(res.chunks_dropped) == 0
    assert not res.is_truncated
    assert "[Evidence Chunk: table_row_0]" in res.context_text
    assert "[Evidence Chunk: text_0]" in res.context_text
    assert "[Evidence Chunk: table_row_1]" in res.context_text
    assert res.token_count > 0


def test_parent_expanded_context_assembly(sample_child_chunks, sample_corpus_index):
    builder = ContextBuilder(default_mode="parent_expanded")
    res = builder.build_context(
        sample_child_chunks[:1], corpus_index=sample_corpus_index
    )

    assert res.mode == "parent_expanded"
    assert len(res.chunks_included) == 1
    assert "=== Parent Document Section: AAPL_2019 ===" in res.context_text
    assert "--- Financial Table ---" in res.context_text
    assert "--- Narrative Context ---" in res.context_text
    assert "[Evidence Chunk: table_row_0]" in res.context_text
    assert "[Evidence Chunk: text_0]" in res.context_text


def test_deduplicated_parent_context_assembly(sample_child_chunks, sample_corpus_index):
    builder = ContextBuilder(default_mode="deduplicated_parent")
    # First two chunks share the same parent AAPL_2019
    res = builder.build_context(
        sample_child_chunks[:2], corpus_index=sample_corpus_index
    )

    assert res.mode == "deduplicated_parent"
    assert len(res.chunks_included) == 1  # Deduplicated down to single parent
    assert res.context_text.count("=== Parent Document Section: AAPL_2019 ===") == 1


def test_token_budget_enforcement(sample_child_chunks):
    # Set a tight token budget that allows only the first chunk
    token_count_first = default_token_counter(format_child_chunk(sample_child_chunks[0]))
    tight_budget = token_count_first + 2

    builder = ContextBuilder(default_mode="child_only", max_tokens=tight_budget)
    res = builder.build_context(sample_child_chunks)

    assert res.is_truncated is True
    assert len(res.chunks_included) == 1
    assert len(res.chunks_dropped) == 2
    assert res.chunks_dropped[0]["chunk_id"] == "text_0"
    assert res.chunks_dropped[1]["chunk_id"] == "table_row_1"
    assert res.truncation_reason is not None
    assert f"budget of {tight_budget}" in res.truncation_reason


def test_atomic_retention_when_single_chunk_exceeds_budget(sample_child_chunks):
    # If budget is ridiculously small (e.g. 5 tokens), first chunk is retained atomically
    builder = ContextBuilder(default_mode="child_only", max_tokens=5)
    res = builder.build_context(sample_child_chunks)

    assert len(res.chunks_included) == 1
    assert len(res.chunks_dropped) == 2
    assert res.is_truncated is True


def test_empty_chunks_returns_empty_result():
    builder = ContextBuilder()
    res = builder.build_context([])

    assert res.context_text == ""
    assert res.chunks_included == []
    assert res.chunks_dropped == []
    assert res.token_count == 0
    assert res.is_truncated is False


def test_missing_corpus_index_raises():
    builder = ContextBuilder(default_mode="parent_expanded")
    with pytest.raises(ValueError, match="corpus_index is required"):
        builder.build_context([{"chunk_id": "c1", "text": "foo"}])


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Invalid default_mode"):
        ContextBuilder(default_mode="invalid_mode")

    builder = ContextBuilder()
    with pytest.raises(ValueError, match="Invalid mode"):
        builder.build_context([{"chunk_id": "c1", "text": "foo"}], mode="bad_mode")


def test_custom_token_counter(sample_child_chunks):
    # Custom counter that counts characters instead of words
    char_counter = len
    builder = ContextBuilder(token_counter=char_counter, max_tokens=100)
    res = builder.build_context(sample_child_chunks)
    assert res.token_count == len(res.context_text)
