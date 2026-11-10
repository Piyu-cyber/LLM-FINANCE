import pytest
from validator import (
    load_corpus,
    build_chunk_lookup,
    validate_claim,
    unit_matches_evidence,
    period_matches_evidence,
    extract_narrative_direction,
    validate_narrative_direction,
)


@pytest.fixture
def sample_lookup():
    chunks = [
        {
            "example_id": "UNP/2009/page_34.pdf-1",
            "chunk_id": "table_1",
            "text": (
                "millions of dollars: interest expense | "
                "2009: -600 ( 600 ) | "
                "2008: -511 ( 511 ) | "
                "2007: -482 ( 482 ) | "
                "% change 2009 v 2008: 17 | % change 2008 v 2007: 6"
            ),
        },
        {
            "example_id": "UNP/2009/page_34.pdf-1",
            "chunk_id": "post_text_0",
            "text": (
                "interest expense increased in 2009 versus 2008 due primarily "
                "to higher weighted-average debt levels."
            ),
        },
    ]
    return build_chunk_lookup(chunks)


def test_extract_narrative_direction():
    text = "Interest expense increased in 2009 versus 2008 due to debt levels."
    res = extract_narrative_direction(text, metric="interest expense")
    assert res["direction"] == "increase"
    assert res["keyword"] == "increased"

    dec_text = "Operating costs decreased significantly during the fiscal period."
    dec_res = extract_narrative_direction(dec_text, metric="operating costs")
    assert dec_res["direction"] == "decrease"
    assert dec_res["keyword"] == "decreased"


def test_validate_narrative_direction_contradiction():
    # Naive signed -89 for an expense vs "increased" narrative text
    evidence_text = [
        "interest expense increased in 2009 versus 2008 due to higher debt."
    ]
    res = validate_narrative_direction(
        claimed_value=-89,
        evidence_texts=evidence_text,
        referent={"entity": "interest expense", "period": "2009", "unit": "million USD"},
    )
    assert res["valid"] is False
    assert res["status"] == "DIRECTIONAL_CONTRADICTION"
    assert "suggested_fix" in res


def test_validate_narrative_direction_economic_match():
    # Economic magnitude +89 matches narrative "increased"
    evidence_text = [
        "interest expense increased in 2009 versus 2008 due to higher debt."
    ]
    res = validate_narrative_direction(
        claimed_value=89,
        evidence_texts=evidence_text,
        referent={"entity": "interest expense", "period": "2009", "unit": "million USD"},
    )
    assert res["valid"] is True
    assert res["status"] == "VERIFIED_DIRECTION"
    assert res["narrative_direction"] == "increase"


def test_validate_claim_with_sample_lookup(sample_lookup):
    claim = {
        "claims": [
            {
                "id": "claim_1",
                "claim_text": "The interest expense in 2009 was 600 million USD.",
                "value": 600,
                "unit": "million USD",
                "citations": ["table_1"],
                "operands": [
                    {
                        "id": "operand_1",
                        "chunk_id": "table_1",
                        "span_text": "2009: -600 ( 600 )",
                        "context_text": "millions of dollars: interest expense | 2009: -600 ( 600 )",
                        "normalized_value": 600,
                        "unit": "million USD",
                        "period": "2009",
                    }
                ],
                "program": [],
                "referent": {
                    "entity": "interest expense",
                    "period": "2009",
                    "unit": "million USD",
                },
                "status": "answered",
            }
        ],
        "final_claim_id": "claim_1",
    }

    result = validate_claim(
        claim,
        sample_lookup,
        example_id="UNP/2009/page_34.pdf-1",
        context_texts=["interest expense increased in 2009 versus 2008"],
    )
    assert result["valid"] is True
    assert result["status"] == "VERIFIED_EVIDENCE"



# ============================================================
# Phase 1: Unit Validation Tests
# ============================================================

def test_unit_matches_evidence_positive_million_usd():
    assert unit_matches_evidence("million USD", "millions of dollars: interest expense | 2009: -600") is True
    assert unit_matches_evidence("million USD", "in millions of dollars") is True
    assert unit_matches_evidence("million USD", "net income was $450 million") is True


def test_unit_matches_evidence_negative_dollar_alone_fails_million_usd():
    assert unit_matches_evidence("million USD", "$600") is False
    assert unit_matches_evidence("million USD", "the cost was $450 total") is False
    assert unit_matches_evidence("million USD", "USD 500") is False


def test_unit_matches_evidence_billion_usd():
    assert unit_matches_evidence("billion USD", "total assets were $12.5 billion") is True
    assert unit_matches_evidence("billion USD", "in billions of dollars") is True
    assert unit_matches_evidence("billion USD", "net income was $450 million") is False
    assert unit_matches_evidence("billion USD", "$12.5") is False


def test_unit_matches_evidence_thousand_usd():
    assert unit_matches_evidence("thousand USD", "operating costs in thousands of dollars") is True
    assert unit_matches_evidence("thousand USD", "in thousands") is True
    assert unit_matches_evidence("thousand USD", "$500") is False


def test_unit_matches_evidence_base_usd():
    assert unit_matches_evidence("USD", "total fee of $50") is True
    assert unit_matches_evidence("USD", "50 USD") is True
    assert unit_matches_evidence("USD", "in millions of dollars: $50") is False


def test_unit_matches_evidence_percentages():
    assert unit_matches_evidence("percent", "growth was 17%") is True
    assert unit_matches_evidence("percent", "increased by 5.4 percent") is True
    assert unit_matches_evidence("percent", "$500 million") is False
    assert unit_matches_evidence("percent", "500 shares") is False


def test_unit_matches_evidence_shares():
    assert unit_matches_evidence("shares", "issued 50,000 common shares") is True
    assert unit_matches_evidence("million shares", "12.5 million shares outstanding") is True
    assert unit_matches_evidence("shares", "$500 million") is False


def test_unit_matches_evidence_per_share():
    assert unit_matches_evidence("USD per share", "diluted EPS of $1.25 per share") is True
    assert unit_matches_evidence("USD per share", "$1.25 / share") is True
    assert unit_matches_evidence("USD per share", "$1.25") is False


# ============================================================
# Phase 1: Fiscal Period Validation Tests
# ============================================================

def test_period_matches_exact_fiscal_years():
    assert period_matches_evidence("2009", "2009: -600 ( 600 )") is True
    assert period_matches_evidence("2008", "for fiscal year 2008") is True
    assert period_matches_evidence("FY2009", "FY2009 revenues") is True
    assert period_matches_evidence("FY 2009", "in FY 2009") is True


def test_period_matches_negative_partial_year():
    assert period_matches_evidence("20", "2009: -600 ( 600 )") is False
    assert period_matches_evidence("09", "page 109 of report") is False
    assert period_matches_evidence("200", "2009: -600") is False
    assert period_matches_evidence("2008", "2009: -600 ( 600 )") is False


def test_period_matches_year_ranges():
    assert period_matches_evidence("2008-2009", "change over 2008-2009 period") is True
    assert period_matches_evidence("2008 to 2009", "growth from 2008 to 2009") is True
    assert period_matches_evidence("2008-2009", "2009 versus 2008 results") is True
    assert period_matches_evidence("2008-2009", "2009 results only") is False


def test_period_matches_quarterly():
    assert period_matches_evidence("Q1 2009", "first quarter Q1 2009 results") is True
    assert period_matches_evidence("Q2 2008", "first quarter Q1 2008 results") is False

if __name__ == "__main__":
    pytest.main([__file__])
