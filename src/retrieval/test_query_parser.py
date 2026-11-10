"""test_query_parser.py — Unit tests for Phase 1 Step 1.2 query parser.

Run with:
    $env:PYTHONPATH="src;src/retrieval;src/verification;src/eval"
    python -m pytest src/retrieval/test_query_parser.py -v

No corpus or network access required.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pytest

from query_parser import (
    extract_years,
    extract_companies,
    parse_query,
    chunk_matches_metadata,
    metadata_boost,
    filter_chunks_by_metadata,
    build_ticker_dict,
)


# ============================================================
# extract_years
# ============================================================

class TestExtractYears:
    def test_single_year(self):
        assert extract_years("What happened in 2009?") == ["2009"]

    def test_two_years_sorted(self):
        assert extract_years("Change from 2008 to 2009?") == ["2008", "2009"]

    def test_duplicate_years_deduplicated(self):
        assert extract_years("2009 vs 2009 fiscal year 2009") == ["2009"]

    def test_no_years(self):
        assert extract_years("What is the revenue?") == []

    def test_year_at_start(self):
        assert extract_years("2015 revenue grew significantly.") == ["2015"]

    def test_year_adjacent_to_text(self):
        assert extract_years("revenue2009") == []  # no word boundary

    def test_range_of_years(self):
        result = extract_years("From 1999 to 2020 the company grew.")
        assert "1999" in result
        assert "2020" in result

    def test_does_not_match_short_numbers(self):
        assert extract_years("99 to 08 comparison") == []


# ============================================================
# extract_companies
# ============================================================

SMALL_TICKER_DICT = {
    "UNP": ["union pacific", "unp"],
    "GS": ["goldman sachs", "goldman", "gs"],
    "JPM": ["jpmorgan", "jp morgan", "jpm"],
}


class TestExtractCompanies:
    def test_full_name_match(self):
        result = extract_companies(
            "What were Union Pacific revenues?",
            SMALL_TICKER_DICT
        )
        assert result == ["UNP"]

    def test_ticker_symbol_match(self):
        result = extract_companies("UNP reported earnings today.", SMALL_TICKER_DICT)
        assert result == ["UNP"]

    def test_case_insensitive_name(self):
        result = extract_companies("GOLDMAN SACHS profits", SMALL_TICKER_DICT)
        assert result == ["GS"]

    def test_multiple_companies(self):
        result = extract_companies(
            "Compare Goldman Sachs and JPMorgan results.",
            SMALL_TICKER_DICT
        )
        assert set(result) == {"GS", "JPM"}

    def test_no_company_match(self):
        result = extract_companies("What is the revenue?", SMALL_TICKER_DICT)
        assert result == []

    def test_partial_name_does_not_match_short_variant(self):
        # "gs" as part of "results" must not match GS
        result = extract_companies("results grew", SMALL_TICKER_DICT)
        assert "GS" not in result

    def test_default_dict_used_when_none(self):
        # Should not raise; uses built-in dict
        result = extract_companies("Union Pacific interest expense")
        assert "UNP" in result


# ============================================================
# parse_query
# ============================================================

class TestParseQuery:
    def test_full_parse(self):
        q = "What is the change in interest expense from 2008 to 2009?"
        result = parse_query(q, SMALL_TICKER_DICT)
        assert result["fiscal_years"] == ["2008", "2009"]
        assert result["has_year"] is True
        assert result["has_company"] is False
        assert result["company_ids"] == []

    def test_with_company(self):
        q = "What were Union Pacific's revenues in 2009?"
        result = parse_query(q, SMALL_TICKER_DICT)
        assert result["company_ids"] == ["UNP"]
        assert result["fiscal_years"] == ["2009"]
        assert result["has_company"] is True
        assert result["has_year"] is True

    def test_empty_question(self):
        result = parse_query("", SMALL_TICKER_DICT)
        assert result["company_ids"] == []
        assert result["fiscal_years"] == []
        assert result["has_company"] is False
        assert result["has_year"] is False

    def test_result_keys_present(self):
        result = parse_query("revenue?", SMALL_TICKER_DICT)
        assert "company_ids" in result
        assert "fiscal_years" in result
        assert "has_company" in result
        assert "has_year" in result


# ============================================================
# chunk_matches_metadata
# ============================================================

SAMPLE_CHUNK = {"example_id": "UNP/2009/page_34.pdf-1", "text": "interest expense"}
GS_CHUNK = {"example_id": "GS/2009/page_5.pdf-1", "text": "revenue"}
UNP_2010_CHUNK = {"example_id": "UNP/2010/page_12.pdf-1", "text": "capex"}


class TestChunkMatchesMetadata:
    def test_company_and_year_match(self):
        assert chunk_matches_metadata(SAMPLE_CHUNK, ["UNP"], ["2009"])

    def test_company_mismatch(self):
        assert not chunk_matches_metadata(GS_CHUNK, ["UNP"], ["2009"])

    def test_year_mismatch(self):
        assert not chunk_matches_metadata(UNP_2010_CHUNK, ["UNP"], ["2009"])

    def test_no_filters_match_all(self):
        assert chunk_matches_metadata(SAMPLE_CHUNK, [], [])
        assert chunk_matches_metadata(GS_CHUNK, [], [])

    def test_company_only_filter(self):
        assert chunk_matches_metadata(SAMPLE_CHUNK, ["UNP"], [])
        assert not chunk_matches_metadata(GS_CHUNK, ["UNP"], [])

    def test_year_only_filter(self):
        assert chunk_matches_metadata(SAMPLE_CHUNK, [], ["2009"])
        assert not chunk_matches_metadata(UNP_2010_CHUNK, [], ["2009"])


# ============================================================
# metadata_boost
# ============================================================

class TestMetadataBoost:
    def test_full_match_gives_max_boost(self):
        boost = metadata_boost(SAMPLE_CHUNK, ["UNP"], ["2009"])
        assert math.isclose(boost, 1.5 * 1.2)

    def test_company_only_match(self):
        boost = metadata_boost(SAMPLE_CHUNK, ["UNP"], [])
        assert math.isclose(boost, 1.5)

    def test_year_only_match(self):
        boost = metadata_boost(SAMPLE_CHUNK, [], ["2009"])
        assert math.isclose(boost, 1.2)

    def test_no_match_gives_one(self):
        boost = metadata_boost(GS_CHUNK, ["UNP"], ["2010"])
        assert math.isclose(boost, 1.0)

    def test_custom_boosts(self):
        boost = metadata_boost(
            SAMPLE_CHUNK,
            ["UNP"],
            ["2009"],
            company_boost=2.0,
            year_boost=1.5,
        )
        assert math.isclose(boost, 2.0 * 1.5)


# ============================================================
# filter_chunks_by_metadata
# ============================================================

class TestFilterChunksByMetadata:
    CORPUS = [SAMPLE_CHUNK, GS_CHUNK, UNP_2010_CHUNK]

    def test_filter_by_company_and_year(self):
        result = filter_chunks_by_metadata(self.CORPUS, ["UNP"], ["2009"])
        assert result == [SAMPLE_CHUNK]

    def test_filter_by_company_only(self):
        result = filter_chunks_by_metadata(self.CORPUS, ["UNP"], [])
        assert SAMPLE_CHUNK in result
        assert UNP_2010_CHUNK in result
        assert GS_CHUNK not in result

    def test_no_filter_returns_all(self):
        result = filter_chunks_by_metadata(self.CORPUS, [], [])
        assert result == self.CORPUS

    def test_empty_result_falls_back_to_all(self):
        # No chunk matches "AAPL/1990"
        result = filter_chunks_by_metadata(
            self.CORPUS, ["AAPL"], ["1990"], fallback_to_all=True
        )
        assert result == self.CORPUS

    def test_empty_result_no_fallback_returns_empty(self):
        result = filter_chunks_by_metadata(
            self.CORPUS, ["AAPL"], ["1990"], fallback_to_all=False
        )
        assert result == []


# ============================================================
# build_ticker_dict (without corpus file — fallback path)
# ============================================================

class TestBuildTickerDict:
    def test_returns_dict_without_corpus(self):
        d = build_ticker_dict(corpus_path=None)
        assert isinstance(d, dict)
        assert "UNP" in d

    def test_missing_file_returns_defaults(self):
        d = build_ticker_dict(corpus_path="/nonexistent/path.jsonl")
        assert "UNP" in d


# ============================================================
# Ambiguous Tickers & Common Words (Phase 1)
# ============================================================

class TestAmbiguousTickers:
    def test_common_words_do_not_trigger_tickers(self):
        # "now" must not match NOW (ServiceNow)
        res = extract_companies("What is the change in interest expense from 2008 to 2009 now?")
        assert "NOW" not in res

        # "low" must not match LOW (Lowe's)
        res = extract_companies("What was the all-time low in gross profit?")
        assert "LOW" not in res

        # "cat" must not match CAT (Caterpillar)
        res = extract_companies("Compare cat and dog performance")
        assert "CAT" not in res

        # "de" must not match DE (Deere)
        res = extract_companies("Risk de-escalation report")
        assert "DE" not in res

    def test_explicit_names_and_uppercase_tickers_match(self):
        # Unambiguous company name variants
        assert "NOW" in extract_companies("What was ServiceNow's 2009 revenue?")
        assert "CAT" in extract_companies("What was Caterpillar's 2009 operating profit?")
        assert "DE" in extract_companies("What were Deere's net sales in 2008?")
        assert "LOW" in extract_companies("What were Lowe's revenues in 2009?")

        # Explicit uppercase tickers in raw query
        assert "LOW" in extract_companies("What were $LOW revenues in 2009?")
        assert "NOW" in extract_companies("What was NOW revenue in 2009?")
        assert "CAT" in extract_companies("What was CAT revenue in 2009?")
